# --------------------------------------------
# Contour-based line following error
# measure algorithm for AGV devices
# ---------------------------------------------
# From: Deniz Çelik - denizcelik2@outlook.com
# --------------------------------------------


def main_control():

    # Required modules

    from time import time
    from copy import copy
    import numpy as np
    import cv2
    import json

    # print version informations
    print(
        f"""Version of Modules\nOpenCV:\t{cv2.__version__}\nNumpy:\t{np.__version__}"""
    )

    def image_read(path):
        """
        Reads image from a "path" for testing purposes.

        Arguments:
        path -- image source path

        Returns:
        img -- image matrix variable (numpy array format)
        """

        image = cv2.imread(path)
        return image

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
        img_sample,
        limit_lower,
        limit_upper,
        size_kernel_morph,
        iter_erosion,
        iter_dilation,
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
        img_morphed = image_morph(
            img_mask, size_kernel_morph, iter_erosion, iter_dilation
        )
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
        img_morphed = cv2.dilate(
            img_eroded, kernel=val_kernel, iterations=iter_dilation
        )
        return img_morphed

    def find_contours(
        img_sample, bin_thresh=25, cnt_title="Contours List", switch_print=True
    ):

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
        bool_thr, img_bin = cv2.threshold(
            img_monoch, bin_thresh, 255, cv2.THRESH_BINARY
        )
        contours, hierarchy = cv2.findContours(
            img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )
        # The contour variable has the same order as the hierarchy variable
        if switch_print:
            list_contours(contours=contours, hierarchy=hierarchy, title=cnt_title)
        return contours, hierarchy

    def list_contours(contours, hierarchy, title="Contours List"):
        """
        Prints all the detected contours for debugging.

        Arguments:
        contours -- list of detected contours
        hierarchy -- list of hierarchical relations of contours
        title -- printing title of detected contours

        Returns:
        void (nothing to return)
        """

        if contours:
            print(title)
            print(
                f"Number of Contours: {len(contours)} (And its hierarchy vals: {len(hierarchy[0])})"
            )
            for i, contour in enumerate(contours):
                val_cy, val_cx, val_area, val_perim = extract_contour(contour)
                print(
                    f"{i} - Tot.Points:{len(contour)}, Area:{val_area} px, Perimeter:{val_perim:.2f}, Center(Y,X):({val_cy},{val_cx}), Hierachy: {hierarchy[0][i]}"
                )
        else:
            print("no contours listed")

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
        mask_boundry_up=0.80,
        error=0,
        bin_thresh=25,
        switch_print=True,
    ):  # to-do: angle formatting

        """
        Computes error from a specific preprocessed contour for main application
        """

        # img_con_test = img_sample.copy()

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

            if switch_print:
                print(
                    f"{len(results_cnt)} of {len(contours)} contours added to results"
                )

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

            final_cnts, final_hier = find_contours(
                img_roi,
                cnt_title="Final Contours List",
                switch_print=switch_cnt_listing,
            )

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

                # cv2.drawContours(img_con_test, final_cnts, -1, (0, 255, 0), 4)
                angles_orienatation = get_orientation(final_cnts)
                print("angles:", angles_orienatation)

                final_results = np.array(final_results)
                final_results[:, 2] = angles_orienatation
                final_results = final_results.tolist()

            if final_results:
                error_mean = mean_error(error_prev)
                selected_x, selected_y, selected_ind = compute_similarity_for_errors(
                    error_mean, final_results
                )
                error_ang = final_results[selected_ind][2]

                # angle centering
                error_ang -= 90

                # cv2.putText(
                #     img_contest,
                #     str(selected_ind),
                #     (80, 300),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 0, 255),
                #     2,
                # )

                forks = recognize_fork(final_results, selected_ind)

            # cv2.imshow("IMG CON TEST", img_con_test)
            print("forks", forks)

            if switch_print:
                print("final results:\n", final_results)

            selected_cnt = final_cnts[selected_ind]

            # image: "selected cnt" for debug mode
            img_blank_select = np.zeros_like(img_sample)
            cv2.drawContours(
                img_blank_select, [selected_cnt], -1, (50, 150, 255), -1, cv2.LINE_4
            )

            error_px = selected_x - val_mid_x

            return (
                results_cnt,
                selected_cnt,
                error_px,
                selected_x,
                selected_y,
                error_ang,
                forks,
                final_results,
                img_blank_cnt,
                img_blank_select,
            )  # boxes

        else:
            pass  # NO CONTOUR PASS

    def mean_error(error_last):

        list_prev_errors[1:] = list_prev_errors[:3]
        list_prev_errors[0] = error_last
        return np.mean(list_prev_errors, dtype=np.int32)

    def compute_similarity_for_errors(error_mean, results):

        results_x = np.array(results)[:, 1]
        dist = np.sqrt((error_mean - results_x) ** 2)
        val_index_min = np.argmin(dist)
        return results_x[val_index_min], results[val_index_min][0], val_index_min

    def get_orientation(contours, img_sample=None):

        # if there are any final contour
        if contours is not None:

            # create stacking slot list
            angles = []
            counter = 0

            # for each contour sample
            for cnt in contours:

                # if the contour has at least 5 points
                if len(cnt) > 5:

                    counter += 1

                    # compute fitting ellipse
                    ellipse = cv2.fitEllipse(cnt)

                    # extract information
                    (x_cen, y_cen), (d1, d2), angle = ellipse

                    # print(xc,yc,d1,d1,angle)
                    # cv2.ellipse(img_copy, ellipse, (0, 255, 0), 3)

                    # compute the major radius
                    radius_major = max(d1, d2) / 2

                    # rectify the angle
                    if angle > 90:
                        angle = angle - 90
                    else:
                        angle = angle + 90

                    if img_sample is not None:
                        x_top = x_cen + np.cos(np.radians(angle)) * radius_major
                        y_top = y_cen + np.sin(np.radians(angle)) * radius_major
                        x_bot = x_cen + np.cos(np.radians(angle + 180)) * radius_major
                        y_bot = y_cen + np.sin(np.radians(angle + 180)) * radius_major
                        cv2.line(
                            img_sample,
                            (int(x_top), int(y_top)),
                            (int(x_bot), int(y_bot)),
                            (0, 0, 255),
                            3,
                        )
                        # cv2.putText(
                        #     img_sample,
                        #     str(angle_pre),
                        #     (80, 80),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     1,
                        #     (0, 255, 0),
                        #     2,
                        # )
                        cv2.putText(
                            img_sample,
                            str(180 - angle),
                            (80, counter * 50 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                    angle = 180 - angle
                    angles.append(angle)

            return angles

        else:
            return None

    def recognize_fork(results, selected_ind):

        # create fork flag variable
        forks = [[False, None], [False, None]]  # right, left

        # define selected_angle by using selected_ind
        selected_angle = results[selected_ind][2]

        if compute_similarity_for_angles(selected_angle, 90, 0.9):

            # for each result sample
            for ind, res in enumerate(results):

                # boolean flag for right turn
                if compute_similarity_for_angles(res[2], 45, 0.8):
                    forks[0] = [True, ind]

                # boolean flag for left turn
                elif compute_similarity_for_angles(res[2], 135, 0.8):
                    forks[1] = [True, ind]

        # return forks flags variable
        return forks

    def compute_similarity_for_angles(angle, reference, threshold=None):

        similarity = min(angle, reference) / max(angle, reference)

        if threshold is not None:
            return similarity > threshold
        else:
            return similarity

    def debug_screen(
        msg,
        img_filtered,
        img_binary,
        img_masked,
        img_contours,
        img_blank_cnt,
        img_blank_slc,
    ):
        if msg == "OFF":
            pass
        elif msg == "ON":
            pass

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

            cv2.namedWindow("image: DEBUG SCREEN", cv2.WINDOW_NORMAL)
            cv2.imshow("image: DEBUG SCREEN", img_debug)
            return img_debug
        else:
            raise ValueError('Invalid Flag for Debug Screen: Use "ON" or "OFF"')

        return None

    def label_center(
        label, center_x, center_y, img_sample, letter_width=14, color=(255, 170, 50)
    ):

        label_len = len(label) * letter_width + letter_width * 2
        label_height = 40
        offset = 25

        y1 = center_y - (label_height + offset)
        x1 = center_x - int(label_len / 2)

        y2 = center_y - offset
        x2 = center_x + int(label_len / 2)

        text_origin = (np.int32(x1 + letter_width), np.int32(y2 - label_height * 0.25))

        black = 0
        img_sample[y1:y2, x1:x2] = cv2.addWeighted(
            img_sample[y1:y2, x1:x2], 0.55, black, 1, 0
        )

        cv2.putText(
            img_sample,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    def render_result(
        img_sample,
        final_cnts,
        fps,
        error_px,
        coor_px,
        coor_py,
        error_ang,
        selected_roi_cnt,
        forks,
        final_results,
    ):

        # create final render window
        cv2.namedWindow("image: RESULT", cv2.WINDOW_NORMAL)

        # create blank image for drawing
        img_blank = np.zeros_like(img_sample)

        # fork detection boolean variables
        left_turn = False
        right_turn = False

        # draw final contours on blank image
        cv2.drawContours(img_blank, final_cnts, -1, (200, 200, 30), -1, cv2.LINE_4)
        # cv2.drawContours(
        #     img_blank,
        #     [selected_roi_cnt],
        #     -1,
        #     (90, 50, 255),
        #     -1,
        #     cv2.LINE_4,  # 50, 150, 255 : 145, 255, 60
        # )

        # add roi window to final render window
        img_sample = cv2.addWeighted(img_sample, 1, img_blank, 0.35, 0)

        # create variable table
        black = 0
        img_sample[:225, :450] = cv2.addWeighted(
            img_sample[:225, :450], 0.55, black, 1, 0
        )

        # write fps to final render window
        cv2.putText(
            img_sample,
            f"FPS:         {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (55, 180, 255),
            thickness=2,
        )

        # write center point of error contour to final render window
        cv2.putText(
            img_sample,
            f"point (x,y):   ({np.int32(coor_px)}, {np.int32(coor_py)})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 170, 50),
            thickness=2,
        )

        # write error value to final render window
        cv2.putText(
            img_sample,
            f"error (px):   {error_px:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (120, 220, 0),
            thickness=2,
        )

        # write error angle to final render window
        cv2.putText(
            img_sample,
            f"angle (deg):  {error_ang:.1f}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (120, 220, 0),
            thickness=2,
        )

        # draw center point circle to final render window
        cv2.circle(
            img_sample,
            (np.int32(coor_px), np.int32(coor_py)),
            8,
            (100, 30, 200),
            -1,
            cv2.LINE_AA,
        )

        label_center(
            "lane_center", np.int32(coor_px), np.int32(coor_py), img_sample, 16
        )

        # draw center point circle to detected fork contour
        if forks[0][0] or forks[1][0]:
            for direction, fork in enumerate(forks):
                if fork[0]:
                    fork_contour_index = fork[1]
                    fork_center_y = np.int32(final_results[fork_contour_index][0])
                    fork_center_x = np.int32(final_results[fork_contour_index][1])

                    fork_str = ""

                    # evaluate direction: 0 = right
                    if direction == 0:
                        right_turn = True
                        fork_str = "right"

                    # evaluate direction: 1 = left
                    elif direction == 1:
                        left_turn = True
                        fork_str = "left"

                    print(f"fork_{fork_str} center:", fork_center_x, fork_center_y)

                    cv2.circle(
                        img_sample,
                        (fork_center_x, fork_center_y),
                        8,
                        (255, 120, 100),
                        -1,
                        cv2.LINE_AA,
                    )

                    label_center(
                        f"fork_{fork_str}", fork_center_x, fork_center_y, img_sample, 15
                    )

        if right_turn:
            right_turn_color = (120, 220, 0)
        else:
            right_turn_color = (100, 0, 255)

        if left_turn:
            left_turn_color = (120, 220, 0)
        else:
            left_turn_color = (100, 0, 255)

        cv2.putText(
            img_sample,
            f"left: {left_turn}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            left_turn_color,
            thickness=2,
        )

        cv2.putText(
            img_sample,
            f"right: {right_turn}",
            (240, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            right_turn_color,
            thickness=2,
        )

        # draw screen mid reference line to final render window
        val_mid_x = int(img_sample.shape[1] * 0.5)
        cv2.line(
            img_sample,
            (val_mid_x, 0),
            (val_mid_x, img_sample.shape[0]),
            (120, 180, 255),
            2,
            cv2.LINE_4,
        )

        # img_shape_y, img_shape_x, ch = img_sample.shape
        # print(img_shape_y, img_shape_x)

        # num_lines_y = 4
        # num_lines_x = 6
        # width_lines_y = int(img_shape_y / num_lines_y)
        # width_lines_x = int(img_shape_x / num_lines_x)
        # print(width_lines_y, width_lines_x)

        # for i in range(1, num_lines_x):
        #     cv2.line(
        #         img_sample,
        #         (i * width_lines_x, 0),
        #         (i * width_lines_x, img_sample.shape[0]),
        #         (120, 180, 255),
        #         1,
        #         cv2.LINE_4,
        #     )
        # for i in range(1, num_lines_y):
        #     cv2.line(
        #         img_sample,
        #         (0, i * width_lines_y),  # (i * width_lines_x, 0)
        #         (img_sample.shape[1], i * width_lines_y),
        #         (120, 180, 255),
        #         1,
        #         cv2.LINE_4,
        #     )

        # show the final render window
        cv2.imshow("image: RESULT", img_sample)  # render

        # return render-processed image
        return img_sample

    def calculate_fps(t2, t1):
        fps = 1 / (t2 - t1)
        print(f"FPS: {fps:.2f}")
        return fps

    def set_timer(seconds):
        pass

    def capture_frame(path_src):

        object_cap = cv2.VideoCapture(path_src)
        return object_cap
        # add try, except block

    def read_parameters(name_json_file):
        with open(name_json_file) as jf_to_read:
            return json.load(jf_to_read)

    def send_error():
        pass

    def create_codec(name_str, img_sample=None, width=None, height=None):

        if width is not None and height is not None:
            video_width = width
            video_height = height

        elif img_sample is not None:
            video_width = img_sample.shape[1]
            video_height = img_sample.shape[0]

        else:
            raise ValueError(
                "at least, one of the img_sample and width/heights parameters should be given"
            )

        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        codec = cv2.VideoWriter(
            f"{name_str}.mp4", fourcc, 24.0, (video_width, video_height)
        )
        return codec

    def save_video_data(codec_object, img_sample):
        codec_object.write(img_sample)

    # VARIABLES PANEL -------------------------------------
    # Some of the variables have a prefix like val_, img_, area_ for simplicity

    # Read parameters from parameters JSON file
    parameters = read_parameters("parameters.json")

    scaler_x = parameters["scaler_x"]
    scaler_y = parameters["scaler_y"]

    # image blurring kernel size
    size_kernel = parameters["size_kernel_blur"]

    # binary thresholding value
    threshold_binary = parameters["threshold_binary"]

    # contour chain approx. coefficient
    approx_chain_coeff = parameters["approx_chain"]

    # HSV masking limits
    lim_hsv_upper = parameters["lim_hsv_upper"]
    lim_hsv_lower = parameters["lim_hsv_lower"]

    # area limit ratios
    ratio_area_upper = parameters["ratio_area_upper"]
    ratio_area_lower = parameters["ratio_area_lower"]

    # morphological process parameters
    size_kernel_morph = parameters["size_kernel_morph"]
    iter_erosion = parameters["iter_erosion"]
    iter_dilation = parameters["iter_dilation"]

    # debug screen switch ("ON" / "OFF"):
    mode_switch = "ON"

    # contours list printing switch
    switch_cnt_listing = False

    # final result printing switch
    switch_final_res = True

    # FPS counter
    val_fps = 0

    # prev error reserve list
    list_prev_errors = [0] * 5

    # input source path
    path = "datas/test-video-1.mp4"

    # INPUT DATA
    object_cap = capture_frame(path)

    # Saving codec
    _, img_codec_sample = object_cap.read()
    img_codec_sample = image_resize(img_codec_sample, scaler_y, scaler_x)
    object_codec = create_codec("output", img_codec_sample)
    # codec for debug
    # object_codec = create_codec("output", height=748, width=1962)

    # ----------------------- CONTROL LOOP ----------------------------------------

    while object_cap.isOpened():

        t1 = time()

        bool_frame, img_sample = object_cap.read()

        if bool_frame:

            # output splitter
            print("-" * 60)

            # image resizing
            img_resized = image_resize(img_sample, scaler_y, scaler_x)
            # print("SHAPE", img_resized.shape)

            # image filtering
            img_filtered = image_filter(img_resized, size_kernel)

            # masking image for contour detection
            (img_masked, img_thresh) = image_mask(
                img_filtered,
                lim_hsv_lower,
                lim_hsv_upper,
                size_kernel_morph,
                iter_erosion,
                iter_dilation,
            )

            # finding contours
            val_contours, val_hierarchy = find_contours(
                img_masked, threshold_binary, switch_print=switch_cnt_listing
            )

            # drawing contours
            img_contours = draw_contours(
                img_masked, val_contours, approx_chain_coeff, val_hierarchy
            )

            # computing error
            (
                val_results_cnt,
                val_selected_cnt,
                val_error_px,
                val_coor_x,
                val_coor_y,
                val_error_ang,
                val_forks,
                val_final_results,
                img_blank_cnt,
                img_blank_slc,
            ) = compute_error(
                img_resized,
                val_contours,
                ratio_area_lower,
                ratio_area_upper,
                bin_thresh=threshold_binary,
                switch_print=switch_final_res,
            )

            # rendering current result
            img_final = render_result(
                img_resized,
                val_results_cnt,
                val_fps,
                val_error_px,
                val_coor_x,
                val_coor_y,
                val_error_ang,
                val_selected_cnt,
                val_forks,
                val_final_results,
            )

            # debug screen option
            img_debug = debug_screen(
                mode_switch,
                img_filtered,
                cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR),
                img_masked,
                img_contours,
                img_blank_cnt,
                img_blank_slc,
            )

            # save_video_data(
            #     object_codec,
            #     img_final,
            # )

            # key interrupt block
            if cv2.waitKey(10) == ord("q"):
                break

            # calculating fps
            t2 = time()
            val_fps = calculate_fps(t2, t1)

        else:
            print("END OF THE SAMPLE")
            break

    cv2.destroyAllWindows()
    object_cap.release()

    # ----------------------- CONTROL LOOP END ------------------------------------


if __name__ == "__main__":
    main_control()
