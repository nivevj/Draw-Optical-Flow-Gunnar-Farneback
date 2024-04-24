import numpy as np
import cv2

def draw_flow_with_text(img, flow, step=16):
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw flow for text regions (Assuming text regions are brighter than background)
    text_mask = (img > 215)
    text_y, text_x = np.where(text_mask)
    for i in range(len(text_x)):
        x1, y1 = text_x[i], text_y[i]
        fx_text, fy_text = flow[y1, x1]
        x2 = int(x1 + fx_text)
        y2 = int(y1 + fy_text)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # # Draw flow for non-text regions
    # y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    # fx, fy = flow[y,x].T
    # lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    # lines = np.int32(lines + 0.5)
    # cv2.polylines(vis, lines, 0, (0, 255, 0))

    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis


# def draw_flow(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
#     fx, fy = flow[y,x].T
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (x2, y2) in lines:
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

#     return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)

    return res

if __name__ == '__main__':
    # Capture video
    video = "input/news_640x360.y4m"
    cap = cv2.VideoCapture(video)
    
    # Read first frame
    ret, prev = cap.read()
    if not ret:
        print("Video not captured")
    
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    # CConfigs
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    # Process frames
    while True:
        ret, img = cap.read()
        if not ret:
            print("Video parsing completed")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Perform Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Update previous
        prevgray = gray
        
        # Draw the flow of the motion tracked
        #cv2.imshow('flow', draw_flow(gray, flow))
        cv2.imshow('flow', draw_flow_with_text(gray, flow))

        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)
        

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print ('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print ('glitch is', ['off', 'on'][show_glitch])
            
    cv2.destroyAllWindows() 			
