# Code starts here:
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from sklearn import cluster
import os


def main():
    if not os.path.exists("results"):
            os.makedirs("results")

    # Function for convolving images
    def convolution(image, kernel):
        image_height, image_width = image.shape  # adding padding
        kernel_height, kernel_width = kernel.shape
        padding_width = int(kernel_width / 2)
        padding_height = int(kernel_height / 2)
        padded_image = np.zeros((image_height + 2 * padding_height, image_width + 2 * padding_width))
        padded_image[padding_height:padded_image.shape[0] - padding_height,
        padding_width:padded_image.shape[1] - padding_width] = image
        padded_image_height, padded_image_width = padded_image.shape
        convolved_matrix = list()
        for i in range(padded_image_height - kernel_height + 1):
            each_row = list()
            for j in range(padded_image_width - kernel_width + 1):
                my_matrix = padded_image[i:i + kernel_height, j:j + kernel_width]
                matrix_product = np.sum(np.multiply(kernel, my_matrix))  # element-wise multiplication and summation
                each_row.append(matrix_product)
            convolved_matrix.append(each_row)
        return np.array(convolved_matrix)

    # Function that outputs a Gaussian kernel
    def get_gaussian_kernel(kernel_size, sigma):
        kernel_u = np.linspace(-np.floor(kernel_size / 2), np.floor(kernel_size / 2), kernel_size)
        denominator = np.sqrt(2 * np.pi) * sigma  # using the separation principle
        for k in range(len(kernel_u)):
            numerator = np.exp(-(kernel_u[k]) ** 2 / (2 * (sigma ** 2)))
            kernel_u[k] = numerator / denominator
        gauss_kernel = np.outer(kernel_u.T, kernel_u.T)
        return gauss_kernel

    # Function that outputs an LOG (Laplacian of Gaussian) kernel
    def get_log_kernel(kernel_size, sigma):
        kernel_u = np.linspace(-np.floor(kernel_size / 2), np.floor(kernel_size / 2), kernel_size)
        kernel_v = np.linspace(-np.floor(kernel_size / 2), np.floor(kernel_size / 2), kernel_size)
        cmpnt_1 = -1 / (np.pi * (sigma ** 4))
        log_kernel = list()
        for u in range(len(kernel_u)):
            log_row = list()
            for v in range(len(kernel_v)):
                cmpnt_2 = 1 - ((kernel_u[u] ** 2 + kernel_v[v] ** 2) / (2 * sigma ** 2))
                cmpnt_3 = np.exp(-(kernel_u[u] ** 2 + kernel_v[v] ** 2) / (2 * sigma ** 2))
                log = cmpnt_1 * cmpnt_2 * cmpnt_3
                log_row.append(log)
            log_kernel.append(log_row)
        return np.array(log_kernel)

    # This function gives an elongated Gaussian kernel, with sigma_x and sigma_y hard-coded
    def get_elongated_gaussian_kernel(kernel_size, sigma):
        sigma_x = sigma
        sigma_y = 3 * sigma_x
        kernel_x = np.linspace(-np.floor(kernel_size / 2), np.floor(kernel_size / 2), kernel_size)
        kernel_y = np.linspace(-np.floor(kernel_size / 2), np.floor(kernel_size / 2), kernel_size)
        denominator_x = np.sqrt(2 * np.pi) * sigma_x
        denominator_y = np.sqrt(2 * np.pi) * sigma_y
        for k in range(len(kernel_x)):
            numerator_x = np.exp(-(kernel_x[k]) ** 2 / (2 * (sigma_x ** 2)))
            kernel_x[k] = numerator_x / denominator_x
            numerator_y = np.exp(-(kernel_y[k]) ** 2 / (2 * (sigma_y ** 2)))
            kernel_y[k] = numerator_y / denominator_y
        e_gauss_kernel = np.outer(kernel_x.T, kernel_y.T)
        return e_gauss_kernel

    # Function that return a Gabor kernel
    def get_gabor_kernel(sigma, theta, lamda, psi, gamma):
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        nstds = 3  # Number on standard deviation sigma
        x_max = max(abs(nstds * sigma_x * np.cos(theta)),
                    abs(nstds * sigma_y * np.sin(theta)))
        x_max = np.ceil(max(1, x_max))
        y_max = max(abs(nstds * sigma_x * np.sin(theta)),
                    abs(nstds * sigma_y * np.cos(theta)))
        y_max = np.ceil(max(1, y_max))
        x_min = -x_max
        y_min = -y_max
        (y, x) = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 /
                           sigma_y ** 2)) * np.cos(2 * np.pi / lamda * x_theta + psi)
        return gb

    # Developing our filter banks
    # Difference of Gaussian filter bank
    def get_dog_filters(scales):
        no_scales = len(scales)
        no_orientations = 16
        orientations = np.linspace(0, 360, no_orientations)
        gradient_horizontal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel kernel

        fig, axs = plt.subplots(no_scales, no_orientations)
        [axi.set_axis_off() for axi in axs.ravel()]

        dog_filters = list()
        for s in range(no_scales):
            gauss_kernel = get_gaussian_kernel(35, scales[s])
            dog_kernel = convolution(gauss_kernel, gradient_horizontal)
            for o in range(no_orientations):
                rotated_filter = ndimage.rotate(dog_kernel, orientations[o], reshape=False)
                dog_filters.append(rotated_filter)
                axs[s, o].imshow(rotated_filter, interpolation='none', cmap='gray')
        fig.savefig('results/dog_fltrs.png')
        plt.close(fig)
        return np.array(dog_filters)

    # Leung-Malik filter bank
    def get_lm_filters(scales):
        gradient_horizontal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).transpose()
        no_scales = len(scales)

        no_orientations = 6
        orientations = np.linspace(0, -180, no_orientations)
        fig, axs = plt.subplots(len(scales), no_orientations * 2)
        [axi.set_axis_off() for axi in axs.ravel()]

        dog_filters = list()
        ddog_filters = list()
        gaussian_filters = list()
        for s in range(no_scales):
            gaussian_kernel = get_gaussian_kernel(35, scales[s])
            gaussian_filters.append(gaussian_kernel)
            if s < 3:
                e_gauss_kernel = get_elongated_gaussian_kernel(35, scales[s])
                dog_kernel = convolution(e_gauss_kernel, gradient_horizontal)
                ddog_kernel = convolution(dog_kernel, gradient_horizontal)
                for o in range(no_orientations):
                    rotated_filter_dog = ndimage.rotate(dog_kernel, orientations[o], reshape=False)
                    rotated_filter_ddog = ndimage.rotate(ddog_kernel, orientations[o], reshape=False)
                    dog_filters.append(rotated_filter_dog)
                    ddog_filters.append(rotated_filter_ddog)
                    axs[s, o].imshow(rotated_filter_dog, interpolation='none', cmap='gray')
                    axs[s, o + 6].imshow(rotated_filter_ddog, interpolation='none', cmap='gray')

        g_scales = [1, np.sqrt(2), 2, 2 * np.sqrt(2), 3, 3 * np.sqrt(2), 6, 6 * np.sqrt(2)]
        log_filters = list()
        for g_s in g_scales:
            log_kernel = get_log_kernel(35, g_s)
            log_filters.append(log_kernel)

        for o in range(no_orientations * 2):
            if o < 8:
                axs[no_scales - 1, o].imshow(log_filters[o], interpolation='none', cmap='gray')
            if o < 4:
                axs[no_scales - 1, o + 8].imshow(gaussian_filters[o], interpolation='none', cmap='gray')

        fig.savefig('results/lm_fltrs.png')
        plt.close(fig)
        lm_filters = dog_filters + ddog_filters + log_filters + gaussian_filters
        return lm_filters

    # Gabor filter bank
    def get_gabor_filters(scales):
        orientations = np.linspace(0, 180, 8)
        no_scales = len(scales)
        no_orientations = len(orientations)
        fig, axs = plt.subplots(no_scales, no_orientations)
        [axi.set_axis_off() for axi in axs.ravel()]
        gabor_kernels = list()
        for s in range(no_scales):
            gauss_kernel = get_gabor_kernel(scales[s], 0, np.pi / 4, 0, 0.25)
            for o in range(no_orientations):
                rotated_filter = ndimage.rotate(gauss_kernel, orientations[o], reshape=False)
                gabor_kernels.append(rotated_filter)
                axs[s, o].imshow(rotated_filter, interpolation='none', cmap='gray')
        fig.savefig('results/gabor_fltrs.png')
        plt.close(fig)
        return gabor_kernels

    # Depth-stacking all the filter responses
    def stack_responses(src_image, filter_bank):
        fltrd_img_0 = cv2.filter2D(src_image, -1, filter_bank[0])
        for f in range(len(filter_bank)):
            if f != 0:
                new_img = cv2.filter2D(src_image, -1, filter_bank[f])
                filter_stack = np.dstack((fltrd_img_0, new_img))
                fltrd_img_0 = filter_stack
        return fltrd_img_0

    # Function to make half-disk masks
    def get_half_disk(matrix_size, no_divisions):
        radius = matrix_size // 2
        # print(radius)
        thetas = np.linspace(-90, 90, no_divisions)
        all_white_points = list()
        for theta in thetas:
            for i in range(radius):
                x = int((radius + 1) - (i + 1) * np.cos(np.radians(theta)))
                y = int((radius + 1) - (i + 1) * np.sin(np.radians(theta)))
                all_white_points.append(tuple([x, y]))
        half_disk_matrix = np.zeros((matrix_size, matrix_size))
        half_disk_matrix[matrix_size // 2, matrix_size // 2] = 1
        for i in range(len(all_white_points)):
            half_disk_matrix[all_white_points[i][0] - 1, all_white_points[i][1] - 1] = 1
        return half_disk_matrix

    def get_half_disk_masks(scales, orientations):
        no_scales = len(scales)
        no_orientations = len(orientations)
        fig, axs = plt.subplots(no_scales, no_orientations)
        [axi.set_axis_off() for axi in axs.ravel()]
        half_disks = list()
        for s in range(no_scales):
            hdisk = get_half_disk(scales[s], 200)
            for o in range(no_orientations):
                rotated_disk = ndimage.rotate(hdisk, orientations[o], reshape=False)
                half_disks.append(rotated_disk)
                axs[s, o].imshow(rotated_disk, interpolation='none', cmap='gray')
        fig.savefig('results/half_disks.png')
        plt.close(fig)
        return half_disks

    # Function to compute the chi-squared distance
    def get_chi_sqrd_dist(mapp, labels, l_disk, r_disk):
        temp = np.zeros((mapp.shape[0], mapp.shape[1]))
        c = np.ones((mapp.shape[0], mapp.shape[1])) * 0.00001
        chi_sqrd_dist = np.zeros((mapp.shape[0], mapp.shape[1]))
        for i in range(max(labels) + 1):
            temp[mapp == i] = 1
            g_i = cv2.filter2D(temp, -1, l_disk)
            h_i = cv2.filter2D(temp, -1, r_disk)
            chi_sqrd_dist += ((g_i - h_i) ** 2) / (g_i + h_i + c)
        return chi_sqrd_dist

    # This function returns the gradient of any map given to it, with its corresponding parameters
    def get_map_gradient(mapp, labels, half_disk_bank):
        temp = mapp
        for i in range(len(half_disk_bank) // 2):
            chi_sqrd_dist = get_chi_sqrd_dist(mapp, labels, half_disk_bank[2 * i], half_disk_bank[2 * i + 1])
            temp = np.dstack((temp, chi_sqrd_dist))
        mean = np.mean(temp, axis=2)
        return mean

    # Executing our program to get the pb-lite output
    def get_pb_lite_image(src_image, canny_image, sobel_image, img_number):
        
        if not os.path.exists("results/"+img_number):
            os.mkdir("results/"+img_number)
        img = src_image

        # Generating our filter bank
        print("Generating our filter bank...")
        dog_scales = [1, 2, 3, 4, 5]
        dog_fltrs = get_dog_filters(dog_scales)

        lms_scls = np.array([1, np.sqrt(2), 2, 2 * np.sqrt(2)])
        lms_fltrs = get_lm_filters(lms_scls)

        lml_scls = np.sqrt(2) * lms_scls
        lml_fltrs = get_lm_filters(lml_scls)

        scls = np.array([3, 5, 7])
        gabor_fltrs = get_gabor_filters(scls)

        # Generating the Texton Map, T
        print("Forming the texton map...")
        dog_stack = stack_responses(img, dog_fltrs)
        lms_stack = stack_responses(img, lms_fltrs)
        lml_stack = stack_responses(img, lml_fltrs)
        gabor_stack = stack_responses(img, gabor_fltrs)
        final_stack = np.dstack((dog_stack, lms_stack, lml_stack, gabor_stack))
        reshaped_image = final_stack.reshape(int(img.shape[0]) * int(img.shape[1]), int(final_stack.shape[2]))
        k_means = cluster.KMeans(n_clusters=64)
        k_means.fit(reshaped_image)
        texture_labels = k_means.predict(reshaped_image)
        texture_map = np.reshape(texture_labels, (img.shape[0], img.shape[1]))
        plt.imsave("results/" + img_number + "/image" + img_number + "_texton.png", texture_map)

        # Brightness map, B
        print("Onto the brightness map...")
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rshaped_image_b = grayscale_image.reshape(int(grayscale_image.shape[0]) * int(grayscale_image.shape[1]), 1)
        k_means = cluster.KMeans(n_clusters=16)
        k_means.fit(rshaped_image_b)
        brightness_labels = k_means.predict(rshaped_image_b)
        b_map = brightness_labels.reshape(int(grayscale_image.shape[0]), int(grayscale_image.shape[1]))
        plt.imsave("results/" + img_number + "/image" + img_number + "_brightness.png", b_map, cmap='gray')

        # Colour map, C
        print("Now, onto the colour map...")
        two_channel_image = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        k_means = cluster.KMeans(n_clusters=16)
        k_means.fit(two_channel_image)
        colour_labels = k_means.predict(two_channel_image)
        c_map = np.reshape(colour_labels, (img.shape[0], img.shape[1]))
        plt.imsave("results/" + img_number + "/image" + img_number + "_color.png", c_map)

        fltr_size = [11, 19, 27]
        disk_orientations = [-90 + 180, -90, -60 + 180, -60, -30 + 180, -30, -15 + 180, -15, 0 + 180, 0, 15 + 180, 15,
                             30 + 180, 30, 60 + 180, 60]
        my_half_disks = get_half_disk_masks(fltr_size, disk_orientations)

        print("Getting started with calculating the gradients...")
        texton_gradient = get_map_gradient(texture_map, list(set(texture_labels)), my_half_disks)
        print("We are now done with the texton map, now onto the brightness map!")
        brightness_gradient = get_map_gradient(b_map, list(set(brightness_labels)), my_half_disks)
        print("We are now done with the brighness map, now onto the colour map!")
        colour_gradient = get_map_gradient(c_map, list(set(colour_labels)), my_half_disks)

        plt.imsave("results/" + img_number + "/image" + img_number + "_texton_grad.png", texton_gradient)
        plt.imsave("results/" + img_number + "/image" + img_number + "_brightness_grad.png", brightness_gradient)
        plt.imsave("results/" + img_number + "/image" + img_number + "_color_grad.png", colour_gradient)

        canny_sobel_product = (1 / 2) * (canny_image) + (1 / 2) * (sobel_image)
        pb_lite_output = np.multiply(((1 / 3) * (texton_gradient + brightness_gradient + colour_gradient)),
                                     canny_sobel_product)
        plt.imsave("results/" + img_number + "/image" + img_number + "_pb_lite.png", pb_lite_output, cmap="gray")
        print("Done with image " + img_number + "!")
        return pb_lite_output

    prnt_drctry = "phase1/BSDS500"
    img_drctry = prnt_drctry + "/Images"
    cny_drctry = prnt_drctry + "/CannyBaseline"
    sbl_drctry = prnt_drctry + "/SobelBaseline"
    for file_name in sorted(os.listdir(img_drctry)):
        if ".jpg" in file_name:
            img_no = file_name.split(".")[0]
            img_file = img_no + ".jpg"
            bsline_file = img_no + ".png"
            print("\nFile being processed: ", os.path.join(img_drctry, img_file))
            ref_img = cv2.imread(os.path.join(img_drctry, img_file))
            cny_img = cv2.imread(os.path.join(cny_drctry, bsline_file), cv2.COLOR_BGR2GRAY)
            sbl_img = cv2.imread(os.path.join(sbl_drctry, bsline_file), cv2.COLOR_BGR2GRAY)
            get_pb_lite_image(ref_img, cny_img, sbl_img, img_no)
    return 0


if __name__ == '__main__':
    main()
