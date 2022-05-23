import cv2
import numpy as np
import json


def nothing(x):
    pass


def image_resize(img_sample, ratio_y, ratio_x):
    """
    Resizes image with specific x and y ratios.

    Arguments:
    img_sample -- input image to resize
    ratio_y -- scale ratio for y-axis
    ratio_x -- scale ratio for x-axis

    Returns:
    image_resized -- resized image matrix variable
    """

    image_resized = cv2.resize(
        img_sample, None, fy=ratio_y, fx=ratio_x, interpolation=cv2.INTER_AREA
    )
    return image_resized


def read_parameters(name_json_file):
    with open(name_json_file) as jf_to_read:
        return json.load(jf_to_read)


def image_filter(img_sample, size_k):
    """
    Applies blur filtering to image.

    Arguments:
    image_sample -- image sample to filtering
    size_k -- filter square kernel size

    Returns:
    image_blurred -- processed output image
    """

    image_blurred = cv2.blur(img_sample, (size_k, size_k))
    return image_blurred


def image_mask(
    img_sample, limit_lower, limit_upper, iter_erosion, iter_dilation, size_kernel_morph
):
    """
    Applies mask in HSV color-space as preprocessing of contour finding.

    Arguments:
    img_sample -- image sample to process
    limit_lower -- HSV color range lower limit
    limit_upper -- HSV color range upper limit

    Returns:
    img_masked -- masked image sample
    img_morphed -- morphology applied image sample from "image_morph" function
    """

    img_hsv = cv2.cvtColor(img_sample, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, np.array(limit_lower), np.array(limit_upper))
    img_morphed = image_morph(img_mask, size_kernel_morph, iter_erosion, iter_dilation)
    img_masked = cv2.bitwise_and(img_sample, img_sample, mask=img_morphed)
    return img_masked, img_morphed


def image_morph(img_sample, size_kernel, iter_erosion, iter_dilation):
    """
    Applies erosion and dilation operators for noise reduction.

    Arguments:
    img_sample -- image sample to process
    size_kernel -- filter kernel size for both filters
    iter_erosion -- number of iterations for erosion filter
    iter_dilation -- number of iterations for dilation filter

    Returns:
    img_morphed -- processed image matrix
    """

    val_kernel = np.ones((size_kernel, size_kernel), np.uint8)
    img_eroded = cv2.erode(img_sample, kernel=val_kernel, iterations=iter_erosion)
    img_morphed = cv2.dilate(img_eroded, kernel=val_kernel, iterations=iter_dilation)
    return img_morphed


def find_contours(img_sample, bin_thresh=25, cnt_title="Contours List"):

    """
    Finds contours on a preprocessed (masked, noise filtered) image.

    Arguments:
    img_sample -- image sample to process
    bin_thresh -- threshold value for binary thresholding
    cnt_title -- printing title for detected contours

    Returns:
    contours -- list of detected contours
    hierarchy -- list of hierarchical relations of contours
    """

    img_monoch = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)
    bool_thr, img_bin = cv2.threshold(img_monoch, bin_thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    # The contour variable has the same order as the hierarchy variable
    # list_contours(contours=contours, hierarchy=hierarchy, title=cnt_title)
    return contours, hierarchy


def debug_screen(
    img_filtered,
    img_binary,
    img_masked,
    img_contours,
    img_blank_cnt,
    img_blank_slc,
):

    img_list = [
        [
            img_filtered,
            img_binary,
            img_masked,
        ],
        [img_contours, img_blank_cnt, img_blank_slc],
    ]

    img_list_resized = [
        [image_resize(img, 0.5, 0.5) for img in img_list_row]
        for img_list_row in img_list
    ]

    val_border_w = int(img_filtered.shape[0] * 0.01)
    img_list_bordered = [
        [
            cv2.copyMakeBorder(
                img,
                val_border_w,
                val_border_w,
                val_border_w,
                val_border_w,
                cv2.BORDER_CONSTANT,
                value=(70, 70, 70),
            )
            for img in img_list_row
        ]
        for img_list_row in img_list_resized
    ]
    img_debug = cv2.vconcat(
        [cv2.hconcat(img_list_row) for img_list_row in img_list_bordered]
    )

    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img_debug)


def draw_contours(img_sample, contours, val_coef_chain, hierarchy=None):
    """
    Draws all the detected contours on input image sample for debugging.

    Arguments:
    img_sample -- image sample to process
    contours -- list of detected contours
    hierarchy -- list of hierarchical relations of contours

    Returns:
    img_sample -- contours drawn image matrix
    """

    img_sample = img_sample.copy()

    if contours:
        for i, contour in enumerate(contours):

            val_epsilon = val_coef_chain * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, val_epsilon, True)

            cv2.drawContours(
                img_sample,
                [contour],
                0,
                (255, 0, 0),
                thickness=5,
                lineType=cv2.LINE_4,
            )
            val_cy, val_cx, val_area, val_perim = extract_contour(contour)
            cv2.putText(
                img_sample,
                f"NO: {i}",
                (val_cx, val_cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (120, 180, 255),
                thickness=2,
            )
            cv2.putText(
                img_sample,
                f"A: {val_area:.2f}",
                (val_cx, val_cy + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (200, 200, 30),
                thickness=2,
            )
            cv2.putText(
                img_sample,
                f"P: {val_perim:.2f}",
                (val_cx, val_cy + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (100, 30, 200),
                thickness=2,
            )
    return img_sample


def extract_contour(cnt_sample):
    """
    Extracts required contour informations.

    Arguments:
    cnt_sample -- contour sample to process from a contours list

    Returns:
    val_cy -- y-axis coordinate of selected contour
    val_cx -- x-axis coordinate of selected contour
    val_area -- total pixel area of selected contour
    val_perim -- perimeter of selected contour
    """

    epsilon = 0.000001

    if cnt_sample.any():
        val_M = cv2.moments(cnt_sample)
        val_cx = int(val_M["m10"] / (val_M["m00"] + epsilon))
        val_cy = int(val_M["m01"] / (val_M["m00"] + epsilon))
        val_area = cv2.contourArea(cnt_sample)
        val_perim = cv2.arcLength(cnt_sample, True)
        return val_cy, val_cx, val_area, val_perim
    return None, None, None, None


def compute_error(
    img_sample,
    contours,
    ratio_lower=0,
    ratio_upper=0,
    mask_boundry_low=0.425,
    mask_boundry_up=0.8,
    error=0,
    bin_thresh=25,
):  # to-do: angle formatting

    """
    Computes error from a specific preprocessed contour for main application
    """

    error_prev = error
    shape_y = img_sample.shape[0]
    shape_x = img_sample.shape[1]
    area_image = shape_y * shape_x
    val_mid_x = shape_x * 0.5
    results_cnt = []
    results_areas = []

    if contours:

        for i, contour in enumerate(contours):

            area_cnt = cv2.contourArea(contour)
            ratio_cnt = area_cnt / area_image

            if ratio_lower < ratio_cnt and ratio_cnt < ratio_upper:
                # add w/h ratio condition
                results_cnt.append(contour)
                results_areas.append(area_cnt)

        # print(f"{len(results_cnt)} of {len(contours)} contours added to results")

        img_blank_cnt = np.zeros_like(img_sample)
        img_blank_mask = img_blank_cnt.copy()

        cv2.drawContours(
            img_blank_cnt,
            results_cnt,
            -1,
            (0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_4,
        )

        val_corners = np.array(
            [
                [
                    (0, int(shape_y * mask_boundry_up)),
                    (shape_x, int(shape_y * mask_boundry_up)),
                    (shape_x, int(shape_y * mask_boundry_low)),
                    (0, int(shape_y * mask_boundry_low)),
                ]
            ]
        )
        cv2.fillPoly(img_blank_mask, val_corners, 255)

        img_blank_mask_gray = cv2.cvtColor(img_blank_mask, cv2.COLOR_BGR2GRAY)

        th, img_field_mask = cv2.threshold(
            img_blank_mask_gray, bin_thresh, 255, cv2.THRESH_BINARY
        )

        img_roi = cv2.bitwise_and(img_blank_cnt, img_blank_cnt, mask=img_field_mask)
        # cv2.imshow("test1", img_roi)

        final_cnts, final_hier = find_contours(img_roi, cnt_title="Final Contours List")

        final_results = []
        # final_boxes = []

        if final_cnts:
            for cont in final_cnts:

                val_cy, val_cx, val_area, val_perim = extract_contour(cont)
                val_box2D = cv2.minAreaRect(cont)
                (cent_x, cent_y), (rect_w, rect_h), angle = val_box2D
                final_results.append(
                    [
                        np.int32(cent_y),
                        np.int32(cent_x),
                        angle,
                        np.int32(val_area),
                        np.int32(val_perim),
                    ]
                )

                # cnt_box = cv2.boxPoints(val_box2D)
                # cnt_box = np.int32(cnt_box)
                # final_boxes.append(cnt_box)

        img_blank_select = np.zeros_like(img_sample)

        if final_results:
            error_mean = mean_error(error_prev)
            selected_x, selected_y, selected_ind = compute_similarity(
                error_mean, final_results
            )
            error_ang = final_results[selected_ind][2]

            # print("FINAL RESULTS:\n", final_results)

            selected_cnt = final_cnts[selected_ind]

            # image: "selected cnt" for debug mode

            cv2.drawContours(
                img_blank_select, [selected_cnt], -1, (50, 150, 255), -1, cv2.LINE_4
            )

            error_px = selected_x - val_mid_x

        else:
            selected_cnt = None
            error_px = None
            selected_x = None
            selected_y = None
            error_ang = None

        # print(
        #     f"""
        # results_cnt {results_cnt}
        # selected_cnt {selected_cnt}
        # error_px {error_px}
        # selected_x {selected_x}
        # selected_y {selected_y}
        # error_ang {error_ang}
        # img_blank_cnt {img_blank_cnt}
        # img_blank_select {img_blank_select}
        # """
        # )

        return (
            results_cnt,
            selected_cnt,
            error_px,
            selected_x,
            selected_y,
            error_ang,
            img_blank_cnt,
            img_blank_select,
        )  # boxes


def mean_error(error_last):
    list_prev_errors[1:] = list_prev_errors[:3]
    list_prev_errors[0] = error_last
    return np.mean(list_prev_errors, dtype=np.int32)


def compute_similarity(error_mean, results):
    results_x = np.array(results)[:, 1]
    dist = np.sqrt((error_mean - results_x) ** 2)
    val_index_min = np.argmin(dist)
    return results_x[val_index_min], results[val_index_min][0], val_index_min


# Read parameters from parameters JSON file
parameters = read_parameters("parameters.json")

scaler_x = parameters["scaler_x"]
scaler_y = parameters["scaler_y"]
size_kernel = parameters["size_kernel_blur"]
threshold_binary = parameters["threshold_binary"]
approx_chain_coeff = parameters["approx_chain"]
lim_hsv_upper = parameters["lim_hsv_upper"]
lim_hsv_lower = parameters["lim_hsv_lower"]
ratio_area_upper = parameters["ratio_area_upper"]
ratio_area_lower = parameters["ratio_area_lower"]
size_kernel_morph = parameters["size_kernel_morph"]
iter_erosion = parameters["iter_erosion"]
iter_dilation = parameters["iter_dilation"]

# prev error reserve list
list_prev_errors = [0] * 5

# Create a window
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar("size_kernel_blur", "image", size_kernel, 50, nothing)
cv2.createTrackbar("thresh_bin", "image", threshold_binary, 255, nothing)
cv2.createTrackbar("HMin", "image", lim_hsv_lower[0], 179, nothing)
cv2.createTrackbar("SMin", "image", lim_hsv_lower[1], 255, nothing)
cv2.createTrackbar("VMin", "image", lim_hsv_lower[2], 255, nothing)
cv2.createTrackbar("HMax", "image", lim_hsv_upper[0], 179, nothing)
cv2.createTrackbar("SMax", "image", lim_hsv_upper[1], 255, nothing)
cv2.createTrackbar("VMax", "image", lim_hsv_upper[2], 255, nothing)
cv2.createTrackbar("chain_appr", "image", int(approx_chain_coeff * 10), 10, nothing)
cv2.createTrackbar("ratio_area_min", "image", int(ratio_area_lower * 100), 100, nothing)
cv2.createTrackbar("ratio_area_max", "image", int(ratio_area_upper * 10), 10, nothing)
cv2.createTrackbar("size_kernel_morph", "image", size_kernel_morph, 50, nothing)
cv2.createTrackbar("iter_erosion", "image", iter_erosion, 15, nothing)
cv2.createTrackbar("iter_dilation", "image", iter_dilation, 15, nothing)

# Initialize HSV min/max values
hMin, sMin, vMin, hMax, sMax, vMax = 0, 0, 0, 0, 0, 0
prev_hMin = prev_sMin = prev_vMin = prev_hMax = prev_sMax = prev_vMax = 0


# Load image
image_org = cv2.imread("datas/new-data-1.png")

# Scale to fixed ratio
height_img = image_org.shape[0]
scaler_y = 1 / (height_img / 720)
scaler_y = round(scaler_y, 3)
scaler_x = scaler_y

img_resized = image_resize(image_org, scaler_y, scaler_x)

while 1:
    # Get current positions of all trackbars
    size_kernel = cv2.getTrackbarPos("size_kernel_blur", "image")
    threshold_binary = cv2.getTrackbarPos("thresh_bin", "image")
    hMin = cv2.getTrackbarPos("HMin", "image")
    sMin = cv2.getTrackbarPos("SMin", "image")
    vMin = cv2.getTrackbarPos("VMin", "image")
    hMax = cv2.getTrackbarPos("HMax", "image")
    sMax = cv2.getTrackbarPos("SMax", "image")
    vMax = cv2.getTrackbarPos("VMax", "image")
    approx_chain_coeff = cv2.getTrackbarPos("chain_appr", "image") / 1000
    ratio_area_lower = cv2.getTrackbarPos("ratio_area_min", "image") / 100
    ratio_area_upper = cv2.getTrackbarPos("ratio_area_max", "image") / 10
    size_kernel_morph = cv2.getTrackbarPos("size_kernel_morph", "image")
    iter_erosion = cv2.getTrackbarPos("iter_erosion", "image")
    iter_dilation = cv2.getTrackbarPos("iter_dilation", "image")

    # Set minimum and maximum HSV values to display
    lim_hsv_lower = [hMin, sMin, vMin]
    lim_hsv_upper = [hMax, sMax, vMax]

    # # image filtering
    if size_kernel > 2:
        img_filtered = image_filter(img_resized, size_kernel)
    else:
        img_filtered = img_resized

    # masking image for contour detection
    (img_masked, img_thresh) = image_mask(
        img_filtered,
        lim_hsv_lower,
        lim_hsv_upper,
        iter_erosion,
        iter_dilation,
        size_kernel_morph,
    )

    # finding contours
    val_contours, val_hierarchy = find_contours(img_masked, threshold_binary)

    # drawing contours
    img_contours = draw_contours(img_masked, val_contours, approx_chain_coeff)

    (
        val_results_cnt,
        val_selected_cnt,
        val_error_px,
        val_coor_x,
        val_coor_y,
        val_error_ang,
        img_blank_cnt,
        img_blank_slc,
    ) = compute_error(img_resized, val_contours, ratio_area_lower, ratio_area_upper)

    # Convert to HSV format and color threshold
    # hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower, upper)
    # result = cv2.bitwise_and(image_resized, image_resized, mask=mask)

    # Print if there is a change in parameter values
    if (
        (prev_hMin != hMin)
        | (prev_sMin != sMin)
        | (prev_vMin != vMin)
        | (prev_hMax != hMax)
        | (prev_sMax != sMax)
        | (prev_vMax != vMax)
    ):
        print(
            "(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)"
            % (hMin, sMin, vMin, hMax, sMax, vMax)
        )
        prev_hMin = hMin
        prev_sMin = sMin
        prev_vMin = vMin
        prev_hMax = hMax
        prev_sMax = sMax
        prev_vMax = vMax

    # # Display result image
    cv2.imshow("image", img_masked)

    debug_screen(
        img_filtered,
        cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR),
        img_masked,
        img_contours,
        img_blank_cnt,
        img_blank_slc,
    )

    if cv2.waitKey(10) & 0xFF == ord("q"):
        print("QUITED WITHOUT SAVING")
        break

    if cv2.waitKey(10) & 0xFF == ord("s"):

        parameters["scaler_x"] = scaler_x
        parameters["scaler_y"] = scaler_y
        parameters["size_kernel_blur"] = size_kernel
        parameters["threshold_binary"] = threshold_binary
        parameters["approx_chain"] = approx_chain_coeff
        parameters["lim_hsv_upper"] = lim_hsv_upper
        parameters["lim_hsv_lower"] = lim_hsv_lower
        parameters["ratio_area_upper"] = ratio_area_upper
        parameters["ratio_area_lower"] = ratio_area_lower
        parameters["size_kernel_morph"] = size_kernel_morph
        parameters["iter_erosion"] = iter_erosion
        parameters["iter_dilation"] = iter_dilation

        # print(
        #     type(scaler_x),
        #     type(scaler_y),
        #     type(size_kernel),
        #     type(threshold_binary),
        #     type(approx_chain_coeff),
        #     type(lim_hsv_upper),
        #     type(lim_hsv_lower),
        #     type(ratio_area_upper),s
        #     type(ratio_area_lower),
        #     type(size_kernel_morph),
        #     type(iter_erosion),
        #     type(iter_dilation),
        # )

        with open("parameters.json", "w", encoding="utf-8") as jf:
            json.dump(parameters, jf, indent=4, ensure_ascii=False)

        print("PARAMETERS SAVED")
        break

cv2.destroyAllWindows()
