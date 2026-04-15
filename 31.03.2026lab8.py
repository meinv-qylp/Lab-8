import math
import time
import cv2


def image_processing():
    img = cv2.imread("variant-7.jpg")

    # 同时翻转（水平+垂直），即旋转180度
    both_flip = cv2.flip(img, -1)

    cv2.imshow('Original', img)
    cv2.imshow('Both Flips (180 deg)', both_flip)

    cv2.imwrite("variant-7_print.jpg", both_flip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_processing():
    # 读取灰度模板并增强对比度
    template = cv2.imread("ref-point.jpg", cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("There is no photo!")
        return

    template = cv2.equalizeHist(template)
    t_h, t_w = template.shape[:2]

    # 打开摄像头（0 为内置摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera!")
        return

    # 加载带透明通道的苍蝇图片
    fly = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED)
    if fly is None:
        print("There's no fly!")
        fly = None

    threshold = 0.45               # 匹配阈值(低)
    down_points = (640, 480)       # 降低分辨率提高处理速度
    frame_counter = 0              # 控制打印频率

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)  # 缩放图像

        # 灰度化 + 直方图均衡化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 模板匹配
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 获取画面中心坐标
        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2
        center_y = frame_h // 2

        if max_val >= threshold:
            # 匹配区域左上角及右下角
            top_left = max_loc
            bottom_right = (top_left[0] + t_w, top_left[1] + t_h)

            # 绘制绿色矩形
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 225), 2)

            # 匹配区域的中心点
            match_cx = top_left[0] + t_w // 2
            match_cy = top_left[1] + t_h // 2
            cv2.circle(frame, (match_cx, match_cy), 5, (0, 0, 255), -1)

            # 叠加苍蝇图片
            if fly is not None:
                fh, fw = fly.shape[:2]
                x1 = match_cx - fw // 2
                y1 = match_cy - fh // 2
                x2 = x1 + fw
                y2 = y1 + fh

                # 确保 ROI 在画面内
                if 0 <= x1 and x2 < frame_w and 0 <= y1 and y2 < frame_h:
                    if fly.shape[2] == 4:          # BGRA 格式
                        b_ch, g_ch, r_ch, alpha = cv2.split(fly)
                        roi = frame[y1:y2, x1:x2]
                        mask = alpha
                        mask_inv = cv2.bitwise_not(mask)

                        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                        fg = cv2.bitwise_and(
                            cv2.merge((b_ch, g_ch, r_ch)),
                            cv2.merge((b_ch, g_ch, r_ch)),
                            mask=mask
                        )
                        frame[y1:y2, x1:x2] = cv2.add(bg, fg)
                    else:
                        frame[y1:y2, x1:x2] = fly

            # 计算到画面中心的欧氏距离
            dx = match_cx - center_x
            dy = match_cy - center_y
            dist = int(math.sqrt(dx * dx + dy * dy))

            # 显示距离
            cv2.putText(
                frame, f"Distance: {dist} px", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

            # 每隔5帧打印一次坐标和距离
            if frame_counter % 5 == 0:
                print(f"Object center: ({match_cx}, {match_cy}), Distance: {dist} px")

        else:
            cv2.putText(
                frame, "No object detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_processing()
    video_processing()