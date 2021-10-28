from PIL import Image, ImageFilter
import argparse
import os
import numpy as np
import cv2

def extract_features(image, first_n=None):
    """Detect and extract features in given image.

    Arguments:
    image -- the image to extract features from,
    first_n -- the number of keypoints to use. Will be the keypoints with the highest response value. Set to None to use all keypoints. (default: None)

    Returns a list of keypoints and a list of descriptors """
    detector = kp_detector = cv2.KAZE_create(upright=True)
    if detector == kp_detector:
        kps, dsc = detector.detectAndCompute(image, None)
    else:
        kps = kp_detector.detect(image)
        kps, dsc = detector.compute(image, kps)
    
    if len(kps) == 0:
        print("no keypoints found!")
        return [],[]
        
    if first_n:
        kd = zip(kps,dsc)
        kd = sorted(kd, key=lambda x: x[0].response, reverse=True)[:first_n]
        kps, dsc = zip(*kd) # unzip

    # if plot:
    #     vis_img1 = None
    #     vis_img1 = cv2.drawKeypoints(image,kps,vis_img1, 
    #                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     plt.imshow(vis_img1)
    #     plt.show()

    return kps, dsc

def plot_template_matches(keypoints_q, keypoints_r, inliers, query_image, reference_image_border, plot_dir):
    import matplotlib.pyplot as plt
    from skimage.feature import plot_matches

    keypoints_q = np.fliplr(keypoints_q)
    keypoints_r = np.fliplr(keypoints_r)
    matches = np.array(list(zip(range(len(keypoints_q)),range(len(keypoints_r)))))

    # print(f"Number of matches: {matches.shape[0]}")
    # print(f"Number of inliers: {inliers.sum()}")
    # fig, ax = plt.subplots(nrows=2, ncols=1)

    # plot_matches(ax[0], (255-query_image), (255-reference_image_border), keypoints_q, keypoints_r,
    #             matches)#,alignment="vertical")
    # plot_matches(ax[1], (255-query_image), (255-reference_image_border), keypoints_q, keypoints_r,
    #             matches[inliers])#,alignment="vertical")
    # # y = query_image.shape[0]
    # # plt.plot([500,1000,1000,500,500],[y,y,0,0,y],"r",linewidth=2)
    # # plt.plot([530,970,970,530,530],[y-30,y-30,30,30,y-30],"g",linewidth=1)
    # # plt.xticks([],[])
    # # plt.yticks([],[])
    # # for spine in ax.spines:
    # #     ax.spines[spine].set_visible(False)
    # plt.savefig(plot_dir+"/template_matches.png")

def estimate_transform(keypoints_q, keypoints_r, query_image, reference_image, plot_dir=None):
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, SimilarityTransform
    #print("number of used keypoints: %d", len(keypoints_q))
    #logging.info("number of matched templates: %d", len(keypoints_r)) # all get matched

    warp_mode_retrieval = "similarity"
    ransac_stop_probability=0.99
    ransac_max_trials=1000
    ransac_random_state=1337

    if warp_mode_retrieval == "affine":
        warp_mode = AffineTransform
    elif warp_mode_retrieval == "similarity":
        warp_mode = SimilarityTransform

    if len(keypoints_r) <= 3:
        return 0, np.eye(3,3) # need to have enough samples for ransac.min_samples. For affine, at least 3
    model, inliers = ransac((keypoints_q, keypoints_r),
                        warp_mode, min_samples=3, stop_probability=ransac_stop_probability,
                        residual_threshold=5, max_trials=ransac_max_trials, random_state=ransac_random_state)

    if inliers is None:
        num_inliers = 0
    else:
        num_inliers = inliers.sum()

    # convert transform matrix to opencv format
    model = model.params
    model = np.linalg.inv(model)
    model = model.astype(np.float32) # opencv.warp doesn't take double

    if plot_dir:
        plot_template_matches(keypoints_q,keypoints_r, inliers, query_image, reference_image, plot_dir)
        # from skimage.transform import warp
        # from matplotlib import pyplot as plt
        # plt.subplot("131")
        # plt.imshow(reference_image)
        # plt.subplot("132")
        # plt.imshow(query_image)
        # plt.subplot("133")
        # y = query_image.shape[0]
        # plt.plot([30,470,470,30,30], [y-30,y-30,30,30,y-30], "g", linewidth=1)
        # image1_warp = warp(query_image, model)
        # plt.imshow(image1_warp)
        # plt.show()

    return num_inliers, model

def ransac_score(pred_img, truth_img, plot_dir=None, downscale_width=500):
    # reduce image size for performance with fixed aspect ratio
    # print("input size", pred_img.shape)
    width, height = (downscale_width, int(pred_img.shape[0]*(downscale_width/pred_img.shape[1])) )
    # print("ransac size", width, height)
    # print(pred_img)
    # print(pred_img.dtype)
    # print(truth_img)
    # cv2.imshow("title",pred_img)
    # cv2.waitKey(-1)
    query_image_small     = cv2.resize(pred_img,  (width,height), interpolation=cv2.INTER_AREA)
    reference_image_small = cv2.resize(truth_img, (width,height), interpolation=cv2.INTER_AREA)

    # extract features from query sheet
    n_descriptors=300
    keypoints_query,     descriptors_query     = extract_features(query_image_small,     first_n=n_descriptors)
    keypoints_reference, descriptors_reference = extract_features(reference_image_small, first_n=n_descriptors)
    # print("num kp q", len(keypoints_query))
    # print("num kp r", len(keypoints_reference))
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # todo: think about crosscheck and matching direction
    matches = bf.match(np.asarray(descriptors_query), np.asarray(descriptors_reference)) # when providing tuples, opencv fails without warning, i.e. returns []
    # print("num matches", len(matches))
    keypoints_q = [keypoints_query[x.queryIdx].pt for x in matches]
    keypoints_r = [keypoints_reference[x.trainIdx].pt for x in matches]#[kp_reference[x.trainIdx] for x in matches]
    # keypoints_r = [[x-config.index_border_train,y-config.index_border_train] for [x,y] in keypoints_r] # remove border from ref images, as they will not be there for registration
    keypoints_q = np.array(keypoints_q)
    keypoints_r = np.array(keypoints_r)
    
    num_inliers, transform_model = estimate_transform(keypoints_q, keypoints_r, query_image_small, reference_image_small, plot_dir)
    # print("num inliers", num_inliers)
    return len(matches), num_inliers

def calc_scores_dir(predictions_dir, truth_dir, plot_dir=None):
    results = []

    for pred_file_name in os.listdir(predictions_dir):
        if not os.path.splitext(pred_file_name)[-1] in [".tif",".png"] or not "_mask" in pred_file_name:
            continue
        print("scoring %s" % pred_file_name)
        truth_file = list(filter(lambda x: (os.path.splitext(pred_file_name.replace("_mask",""))[0] +"_mask" == os.path.splitext(x)[0]) , os.listdir(truth_dir)))
        if len(truth_file) == 0:
            print(pred_file_name, "couldn't find truth mask!")
            continue
        pred_file = predictions_dir + "/" + pred_file_name
        truth_label = os.path.splitext(truth_file[0])[0].replace("_mask","")
        truth_file = truth_dir + "/" + truth_file[0]

        pred_img = Image.open(pred_file)
        truth_img = Image.open(truth_file).crop((0,0,*pred_img.size))
        # hack: alleviate rescaling effects
        # pred_img = pred_img.filter(ImageFilter.MinFilter(7))
        # truth_img = truth_img.filter(ImageFilter.MaxFilter(7))
        pred_array = np.array(pred_img, dtype=np.int32)
        if len(pred_array.shape)==3:
            pred_array = pred_array[:,:,0] # prediction has 3 channels, but they are all the same
        truth_array = np.array(truth_img, dtype=np.int32)
        intersection = np.sum(np.logical_and(pred_array,truth_array))
        union = np.sum(np.logical_or(pred_array,truth_array))
        iou = intersection/union
        dice = 2*iou/(1+iou)
        # print("IoU:", iou)
        # print("Dice:", dice)

        if plot_dir:
            from matplotlib import pyplot as plt
            difference = np.zeros((*pred_array.shape,3), dtype=np.int32)
            difference[:,:,0] = pred_array#np.subtract(pred_array, truth_array)
            difference[:,:,1] = truth_array#np.subtract(truth_array, pred_array)
            difference[:,:,2] = (truth_array & pred_array)
            plt.imshow(difference)
            fpArtist = plt.Line2D((0,1),(0,0), color='r')
            fnArtist = plt.Line2D((0,1),(0,0), color='g')
            mArtist = plt.Line2D((0,1),(0,0), color='w')
            plt.legend([fpArtist,fnArtist,mArtist], ['FP', 'FN', "match"])
            plt.savefig("%s/%s_fpfn.png" % (plot_dir,os.path.splitext(pred_file_name)[0]))
            # plt.show()
            plt.close()
        
        # ransac score as with MaRE
        pred_img = np.array(pred_img)
        if len(pred_img.shape)==3:
            pred_img = pred_img[:,:,0] # prediction has 3 channels, but they are all the same
        num_matches, ransac_score_result = ransac_score( pred_img, np.array(truth_img), plot_dir=plot_dir)

        # todo: can we get an index score as well? train index on all quads?
        from mare.indexing import search_mask_in_index
        index_rank = search_mask_in_index(pred_img, truth_label)

        result = {  "file": pred_file_name,
                    "iou": iou,
                    "dice": dice,
                    "index rank": index_rank,
                    "num matches": num_matches,
                    "ransac": ransac_score_result }
        results.append(result)
        print(result)
        # print("%s,%0.3f,%0.3f,%d,%d" % (pred_file_name, iou, dice, num_matches, ransac_score_result))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate scores for a full prediction image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prediction', help='directory with prediction images')
    parser.add_argument('truth', help='directory with coresponding truth masks')
    parser.add_argument('--plot', help='plot prediction difference', default=None)
    args = parser.parse_args()
    # python score_predictions.py "/e/experiments/deepseg_exp/26/" "/e/data/usgs/100k/selected/test/"

    calc_scores_dir(args.prediction, args.truth, args.plot)