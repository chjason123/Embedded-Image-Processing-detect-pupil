# -*- coding: utf-8 -*-
"""
Created on Sat Aug   3 12:46:13 2019
@author: LALIT ARORA (Modified for Dual Pupil Detection & Batch Processing)
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
        
        # 建立一個列表來儲存所有輪廓及其與重心的距離
        cnt_with_distance = []
        
        for cnt in cnts:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            # 排除太小或過大的雜訊輪廓（可根據你的圖片解析度調整範圍）
            if radius < 3 or radius > 100: 
                continue
                
            distance = abs(self._centroid[0] - x) + abs(self._centroid[1] - y)
            cnt_with_distance.append((cnt, distance))
        
        # 依據與重心的距離「由小到大」排序
        cnt_with_distance.sort(key=lambda item: item[1])
        
        # 取出前 2 個最接近重心的輪廓（即左眼與右眼）
        final_cnts = [item[0] for item in cnt_with_distance[:2]]
        
        # 迭代繪製這兩個瞳孔
        self._pupils = []
        for i, cnt in enumerate(final_cnts):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 在原圖上畫出檢測到的瞳孔（用藍色圓圈標記）
            cv2.circle(self._img, center, radius, (255, 0, 0), 2)
            # 順便標註這是第幾個偵測到的瞳孔
            cv2.putText(self._img, f"Pupil {i+1}", (center[0]-20, center[1]-radius-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self._pupils.append((center[0], center[1], radius))
        
        filename = os.path.basename(self._img_path)
        print(f"[{filename}] 成功偵測到 {len(self._pupils)} 個瞳孔：", self._pupils)
        
        # 如果有指定儲存路徑，則寫入檔案
        if save_path:
            cv2.imwrite(save_path, self._img)
        else:
            # 若無指定儲存路徑，才彈出視窗（避免批次處理時視窗卡住）
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
    output_folder = r'Eye Images Output' 
    
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
        # 檢查是否為圖片檔案
        if file.lower().endswith(valid_extensions):
            img_count += 1
            full_input_path = os.path.join(input_folder, file)
            full_output_path = os.path.join(output_folder, f"processed_{file}") # 新檔名前面加上 processed_
            
            # 實例化並執行偵測，傳入儲存路徑
            detector = pupil_detection(full_input_path)
            detector.start_detection(save_path=full_output_path)
            
    print(f"\n批次處理完成！共處理了 {img_count} 張圖片，結果已儲存至：{output_folder}")