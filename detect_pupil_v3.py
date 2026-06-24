# -*- coding: utf-8 -*-
"""
Created on Sat Aug   3 12:46:13 2019
@author: LALIT ARORA (Modified for Dual Pupil Detection, Distance Constraint & Visual Annotations)
"""

import cv2
import numpy as np
import os

class pupil_detection():
    def __init__(self, image_path):
        '''
        initialize the class and set the class attributes
        '''
        self._img = None
        self._img_path = image_path
        self._pupils = [] # 修改為儲存多個瞳孔資料的列表
        self._centroid = None
        
    def load_image(self):
        '''
        load the image based on the path passed to the class
        '''
        self._img = cv2.imread(self._img_path)
        if self._img is None:
            return False
        else:
            return True
    
    def show_image (self, img):
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def centroid (self):
        # convert image to grayscale image
        gray_image = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
        # calculate moments of binary image
        M = cv2.moments(thresh)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        self._centroid = (cX, cY)
        
    def detect_pupil (self, save_path=None):
        dst = cv2.fastNlMeansDenoisingColored(self._img, None, 10, 10, 7, 21)
        blur = cv2.GaussianBlur(dst, (5, 5), 0)
        inv = cv2.bitwise_not(blur)
        thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations = 1)
        ret, thresh1 = cv2.threshold(erosion, 210, 255, cv2.THRESH_BINARY)
        cnts, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 建立一個列表來儲存所有輪廓及其與重心的距離與中心座標
        cnt_with_distance = []
        
        for cnt in cnts:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            # 排除太小或過大的雜訊輪廓
            if radius < 3 or radius > 100: 
                continue
                
            distance = abs(self._centroid[0] - x) + abs(self._centroid[1] - y)
            cnt_with_distance.append((cnt, distance, (x, y)))
        
        # 依據與重心的距離「由小到大」排序
        cnt_with_distance.sort(key=lambda item: item[1])
        
        final_cnts = []
        selected_centers = []
        first_pupil_diameter = 0
        
        for item in cnt_with_distance:
            current_cnt = item[0]
            current_center = item[2]
            
            # 計算當前輪廓的半徑與直徑（保留浮點數精確度，先不轉 int）
            _, current_radius = cv2.minEnclosingCircle(current_cnt)
            current_diameter = current_radius * 2
            
            # 如果是第一個選中的瞳孔（基準眼）
            if len(final_cnts) == 0:
                final_cnts.append(current_cnt)
                selected_centers.append(current_center)
                first_pupil_diameter = current_diameter
                continue
            
            # 如果已經有第一個瞳孔了，檢查第二個候選輪廓
            is_valid_pupil = True
            for prev_center in selected_centers:
                # 1. 計算兩點間的總體幾何距離 (歐幾里得距離)
                dist_between_pupils = np.sqrt((current_center[0] - prev_center[0])**2 + 
                                              (current_center[1] - prev_center[1])**2)
                
                # 2. 計算兩點間的垂直距離 (Y 軸差值)
                vertical_distance = abs(current_center[1] - prev_center[1])
                
                # 3. 計算兩個瞳孔的直徑誤差絕對值
                diameter_diff = abs(current_diameter - first_pupil_diameter)
                
                # 【約束條件 1】防重疊：兩眼中心總體距離必須大於第一個瞳孔直徑的 6.5 倍
                if dist_between_pupils < (first_pupil_diameter * 6.5):
                    is_valid_pupil = False
                    break
                    
                # 【約束條件 2】垂直限制：垂直高度差不能大於第一個瞳孔的直徑
                if vertical_distance > first_pupil_diameter:
                    is_valid_pupil = False
                    break
                
                # 【約束條件 3】大小相近限制（二選一，目前採用推薦的 10% 彈性比例）：
                # 方案 A (依你要求): diameter_diff > 0.5 （太嚴格，容易誤殺）
                # 方案 B (實務推薦): diameter_diff > (first_pupil_diameter * 0.1) （容許 10% 誤差）
                if diameter_diff > (first_pupil_diameter * 0.1):
                    is_valid_pupil = False
                    break
            
            if is_valid_pupil:
                final_cnts.append(current_cnt)
                selected_centers.append(current_center)
            
            if len(final_cnts) == 2:
                break
        
        # 迭代繪製這兩個瞳孔
        self._pupils = []
        centers_for_line = []
        
        for i, cnt in enumerate(final_cnts):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            diameter = radius * 2
            centers_for_line.append(center)
            
            # 在原圖上畫出檢測到的瞳孔（用藍色圓圈標記）
            cv2.circle(self._img, center, radius, (255, 0, 0), 2)
            # 標註瞳孔編號與直徑大小
            cv2.putText(self._img, f"P{i+1} Dia:{diameter}px", (center[0]-40, center[1]-radius-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            self._pupils.append((center[0], center[1], radius))
        
        # 如果成功偵測到兩個瞳孔，繪製兩眼連線與距離標示
        if len(centers_for_line) == 2:
            pt1, pt2 = centers_for_line[0], centers_for_line[1]
            pd_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            # 畫一線段連接兩瞳孔中心 (綠色)
            cv2.line(self._img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            
            # 計算連線的中心點，用來放距離文字
            mid_point = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2) - 15)
            cv2.putText(self._img, f"Dist: {pd_distance:.1f}px", mid_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        filename = os.path.basename(self._img_path)
        print(f"[{filename}] 成功偵測到 {len(self._pupils)} 個瞳孔：", self._pupils)
        
        # 如果有指定儲存路徑，則寫入檔案
        if save_path:
            cv2.imwrite(save_path, self._img)
        else:
            self.show_image(self._img)
        
    def start_detection(self, save_path=None):
        if self.load_image():
            self.centroid()
            self.detect_pupil(save_path)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')

# ==================== 批次執行區域 ====================
if __name__ == "__main__":
    # 1. 設定你的輸入與輸出資料夾路徑
    input_folder = r'Eye Images' 
    output_folder = r'Eye Images Output version 3' 
    
    # 2. 如果輸出資料夾不存在，自動建立它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已建立輸出資料夾：{output_folder}")
        
    # 3. 支援的圖片副檔名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 4. 開始遍歷資料夾內的所有檔案
    files = os.listdir(input_folder)
    img_count = 0
    
    for file in files:
        if file.lower().endswith(valid_extensions):
            img_count += 1
            full_input_path = os.path.join(input_folder, file)
            full_output_path = os.path.join(output_folder, f"processed_{file}")
            
            detector = pupil_detection(full_input_path)
            detector.start_detection(save_path=full_output_path)
            
    print(f"\n批次處理完成！共處理了 {img_count} 張圖片，結果已儲存至：{output_folder}")