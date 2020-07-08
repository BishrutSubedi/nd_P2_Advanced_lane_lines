import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
# %matplotlib qt

# Module Camera calibration: Needs to be done once
def camera_Calibration():
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            
            # Calibrating the camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

# Module Color/Gradient Thresholding
def color_gradient_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # img = np.float32(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary



def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9 
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    # This indices are appended for each small window
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(9): # Going through only one window
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial_rect(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    #declare a global variable
    global left_fit
    global right_fit

    global left_fitx
    global right_fitx

    global ploty
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #generate the coeffecient of line using left and right indices of the fit
    # print(type(left_fit))
    # print(left_fit.ndim)a
    # print("left_fit:" + str(left_fit))
    # print("right_fit:" + str(right_fit))

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img


def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]) #non-zero indices in y direction
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ( (nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)) ) 
                    # (X > Ay^2 + BY + C - 100) and (X < Ay^2 + BY + C + 100) 
                    # finding index of left lane pixel within this range
                    # index are linear and increases linearly unlike pixel (can get pixel from index) 
                    # gives both x and y index

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
                    # (X > Ay^2 + BY + C - 100) and (X < Ay^2 + BY + C + 100) 
                    # finding index rigt lane pixel within this range
    
    # Again, extract left and right line pixel positions
    #nonxzerox and nonzeroy have all activated pixels, extracting only pixel indices within range from activated pixels
    leftx = nonzerox[left_lane_inds]   
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    # returns left x, y and rightx, y line.

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right li e pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))


    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines  onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result



def fit_poly(img_shape, leftx, lefty, rightx, righty):
     
     global left_fit
     global right_fit

     global left_fitx
     global right_fitx

     global ploty
          ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
     left_fit = np.polyfit(lefty, leftx, 2)      #returns a , b and c for polynomial
     right_fit = np.polyfit(righty, rightx, 2)
     # Generate x and y values for plotting
     ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
     ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]      # a(y^2) + b y + c
     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
     
     return left_fitx, right_fitx, ploty

def draw_lines(warped):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result
    # plt.imshow(result)

def measure_curvature_rad(left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/675 # meters per pixel in y dimension
    xm_per_pix = 3.7/812 # meters per pixel in x dimension

    y_generated = np.linspace(0, image.shape[0]-1, image.shape[0] )

    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(y_generated)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    radius_average = round(( (left_curverad + right_curverad) / 2000 ) , 4) 

    return radius_average


def addText(image, org, text):
    # text = "Radius of curvature: " + str(radius) +"Km"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return image

def laneOffset():
    xm_per_pix = 3.7/812
    x_center = image.shape[1] / 2
    y = image.shape[0]

    left_linex = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]      # a(y^2) + b y + c
    right_linex = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

    lane_center = left_linex + ( (right_linex - left_linex) /  2 )
    l_offset = (xm_per_pix * abs(x_center - lane_center) )
    l_offset = round( l_offset, 4)
    return l_offset

# Main code

start_time = time.time()

#global variables to store array coeffecient
# These coeffecient will be filled from np.polyfit and 1st sliding window
# to pass to search_poly() to find the non-zero pixels around the margin

left_fit = np.float32(np.zeros([3]))  #create 0 1x3 array in numpy
right_fit = np.float32(np.zeros([3]))

# Empty initializes to random and computes result in search around poly, not nice use np.zero
# left_fit = np.float32(np.empty([3]))  #create empty 1x3 array in numpy
# right_fit = np.float32(np.empty([3]))


# print(left_fit)   #Outputs zero
# print(right_fit)  #Outputs zero

#Calibrating the camera from given chess board images
ret, mtx, dist, rvecs, tvecs = camera_Calibration()

# Load new images of the road
# image = cv2.imread('straight_lines1.jpg')
# image = cv2.imread('straight_lines2.jpg')
# image = cv2.imread('test2.jpg')
# cv2.imshow('',image)

# Global variable for image processing
count = 0

cap = cv2.VideoCapture('test_videos/project_video.mp4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('Output_video/test_output_syl.avi',fourcc, 20.0, (frame_width,frame_height))
out = cv2.VideoWriter('Output_video/test_output1.avi',fourcc, 20.0, (frame_width,frame_height))

print("\nVideo Processing started...\n")

while(cap.isOpened()):
    
    ret,image = cap.read()
    if ret == True:
        count = count +1 # Frame counter

        #Correcting for distorting using matrix from camera calibrateion
        undist = cv2.undistort(image, mtx, dist, None, mtx)

        # Converting the undistorted image to gray scale
        gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)

        # color/gradient threshold
        color_binary,combined_binary =  color_gradient_threshold(undist) 

        ## Getting perspective transform for each images
        # offset = 100

        #extract image shape
        img_size = (gray.shape[1], gray.shape[0])

        # src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        src = np.float32([[755,495], [1050,685], [255,685],[530,495]])

        # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([[1050,495], [1050,685], [255,685],[255,495]])

        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src,dst)    

        # Finding inverse perspective transform
        Minv = cv2.getPerspectiveTransform(dst, src)

        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(combined_binary, M, img_size)       

        # May look redundant, but this function has global variable updates in it. so keep it
        # Try returning only the left/right fit and check it with searchpoly()
        # Note: HAVE TO CALL fit_polynomial_rect(warped) at least once as it updates global variable for search_around_poly

        if count == 1:
            out_img = fit_polynomial_rect(warped) #updates the global variables left_fit/right_fit
        else:
            result = search_around_poly(warped)

        radius_average = measure_curvature_rad(left_fit, right_fit)
        l_Offset = laneOffset() #This fn takes global variable

        #draw rectangle, lines, and text
        result = draw_lines(warped)
        result = addText(result, (50,50) , "Radius of curvature: " + str(radius_average) +"Km")
        result = addText(result, (50, 100), "Lane offset: " + str(l_Offset)+"m")

        out.write(result)

        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break 
  # Break the loop 
    else:
        break  

print("Video Processing complete.")
cap.release() 
out.release()
   
# Closes all the frames 
cv2.destroyAllWindows() 

print("Time taken: %s s" % (time.time() - start_time))
