/** @file pallet_seg_pose.cpp
  * @brief 使用SOLOv2托盤遮罩內的點雲，估測托盤姿態

  * 輸入：(1)/camera/depth_registered/points RGBD相機之有序點雲
         (2)/solo_mask SOLOv2遮罩
  * 輸出：(1)/pallet_cloud_pub 遮罩內提取的點雲 
  *      (2)/pallet_pose 托盤姿態
  * organized_cloud_cb：轉換有序點雲格式
  * pallet_mask_cb：讀取SOLOv2遮罩
  * get_multi_pallet_cloud：針對多個遮罩
  *      (1)提取遮罩內點雲 (2)est_scene_plane_coeff估測姿態 (3)發佈托盤點雲&姿態
  * est_scene_plane_coeff：以SAC_RANSAC擬合遮罩提取出的托盤點雲，
  *      擬合之平面法向量即為托盤姿態
*/
// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pallet_seg/pallet_mask.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

//opencv
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// boost
#include <boost/make_shared.hpp>

//C++ 
#include <vector>
#include <iostream>
#include <algorithm>

typedef pcl::PointXYZRGB PointTRGB;
typedef boost::shared_ptr<pcl::PointCloud<PointTRGB>> PointCloudTRGBPtr;

using namespace std;

struct Center2D
{
    int x;
    int y;
};

struct Center3D
{
    float x;
    float y;
    float z;
};

struct Box2D
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

struct Pallet
{
    cv_bridge::CvImagePtr pallet_mask_ptr;
    std::string pallet_class;
    float probability;
    Box2D box_pixel;
    Center2D center_pixel;
    Center3D center_point;
    PointCloudTRGBPtr depth_cloud;
};

//=====Parameters Setting=====//
bool input_cloud_from_file = false;
std::string file_path_cloud_depth = "./src/pallet_seg_pose/organized_cloud_tmp.pcd";
bool show_plane_remain = false;       //true:顯示RANSAC擬合之托盤平面＆平面外剩餘點雲; false：不顯示
bool show_extracted_cloud = false;    //true:顯示從邊界框提取之點雲; false：最終擬合到邊界框提取之點雲的平面點
//=====Parameters Setting=====//

std::vector<std::string> pallet_type{};
std::vector<Pallet> pallet_all{};

int img_width;
int img_height;

cv_bridge::CvImagePtr solo_mask_ptr;
pcl::PointCloud<PointTRGB>::Ptr organized_cloud_ori(new pcl::PointCloud<PointTRGB>);
pcl::PointCloud<PointTRGB>::Ptr forking_pt(new pcl::PointCloud<PointTRGB>);

ros::Publisher pallet_cloud_pub, multi_pallet_cloud_pub, pallet_pose_pub, fork_pt_pub;
sensor_msgs::PointCloud2 pallet_cloud_msg, multi_pallet_cloud_msg, forking_pt_msg;

void est_scene_plane_coeff(pcl::PointCloud<PointTRGB>::Ptr pallet_cloud_ori, pcl::ModelCoefficients::Ptr coeff_plane, pcl::PointCloud<PointTRGB>::Ptr pallet_plane)
{
    pallet_plane->clear();
    pcl::PointCloud<PointTRGB>::Ptr pallet_plane_outlier(new pcl::PointCloud<PointTRGB>);
    pcl::PointIndices::Ptr inlier_plane(new pcl::PointIndices);

    pcl::SACSegmentation<PointTRGB> seg_plane;
    seg_plane.setOptimizeCoefficients(true);
    seg_plane.setModelType(pcl::SACMODEL_PLANE);
    seg_plane.setMethodType(pcl::SAC_RANSAC);
    seg_plane.setDistanceThreshold(0.030);      //0.03m內的點雲
    seg_plane.setInputCloud(pallet_cloud_ori);
    seg_plane.segment(*inlier_plane, *coeff_plane);

    // std::cerr << "Model inliers: " << inlier_plane->indices.size () << std::endl;
    // for (std::size_t idx = 0; idx < inlier_plane->indices.size (); ++idx)
    //     pallet_plane->push_back(pallet_cloud_ori->points[idx]);

    pcl::ExtractIndices<PointTRGB> extract;
    extract.setInputCloud(pallet_cloud_ori);
    extract.setIndices(inlier_plane);
    extract.setNegative(false);
    extract.filter(*pallet_plane);
    extract.setNegative(true);
    extract.filter(*pallet_plane_outlier);
    std::cerr << "points (ori, plane, outlier): " 
            << pallet_cloud_ori->size() <<", "
            << pallet_plane->size() <<", "
            << pallet_plane_outlier->size() << std::endl;

    // === Use PCL_VISUALIZER to check Segmentation Result ===//
    if(show_plane_remain==true)
    {
        //Draw arrow to display pallet pose
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*pallet_plane, centroid);
        Eigen::Vector3f vv = Eigen::Vector3f(coeff_plane->values[0], coeff_plane->values[1], coeff_plane->values[2]);
        vv.normalize();

        PointTRGB a, b;
        a.x = centroid[0];
        a.y = centroid[1];
        a.z = centroid[2];
        b.x = centroid[0] + 0.3*vv[0];
        b.y = centroid[1] + 0.3*vv[1];
        b.z = centroid[2] + 0.3*vv[2];
        cout << "a:" << a << endl;
        cout << "b:" << b << endl;

        pcl::visualization::PCLVisualizer::Ptr view(new pcl::visualization::PCLVisualizer("pallet_cloud_ori plane viewer"));
        view->removeAllPointClouds();
        view->setBackgroundColor(0, 0, 0);
        view->addCoordinateSystem(0.3f);

        pcl::visualization::PointCloudColorHandlerCustom<PointTRGB> plane_color(pallet_plane, 0, 255, 0); // green
        pcl::visualization::PointCloudColorHandlerCustom<PointTRGB> outlier_color(pallet_plane_outlier, 255, 0, 0); // green
        view->addPointCloud<PointTRGB>(pallet_plane, plane_color, "plane",0);
        view->addPointCloud<PointTRGB>(pallet_plane_outlier, outlier_color, "pallet_plane_outlier", 0);
        view->addArrow<PointTRGB>(b, a, 255, 255, 0, "arrow");
        view->addPlane(*coeff_plane, "inliers");
        view->spin();
    }
    // === Use PCL_VISUALIZER to check Segmentation Result ===//
}

void CalculatePCA(pcl::PointCloud<PointTRGB>::Ptr & cloud, Eigen::Matrix3f& eigenVectorsPCA)
{
    Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	
    eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    
	// eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //校正主方向间垂直
	// eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	// eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
 
	// std::cout << "特徵值 Eigenvalue va(3x1):\n" << eigenValuesPCA << std::endl;
	// std::cout << "特徵向量 Eigenvector ve(3x3):\n" << eigenVectorsPCA << std::endl;
	// std::cout << "質心點(4x1):\n" << pcaCentroid << std::endl;
}

bool get_multi_pallet_cloud()
{
    if(organized_cloud_ori->size()==0)
    {
        cout << "no organized_cloud_ori" << endl;
        return false;
    }
    else
        cout << "[organized_cloud_ori] total points= " << organized_cloud_ori->size() << endl;

    cout << "\n\n\n"
            << "//===============================//\n"
            << "**Get MultiPallet Cloud, START!**\n"
            << "//===============================//\n\n";

    //========================================//
    // Extract Multi Pallet Cloud using SOLOv2 Mask
    // save 2D, 3D sauce information
    //========================================//
    int rows = pallet_all[0].pallet_mask_ptr->image.rows;
    int cols = pallet_all[0].pallet_mask_ptr->image.cols;
    cout << "image (rows, cols) = " << rows << ", " << cols << endl;
    cout << "cloud (rows, cols) = " << organized_cloud_ori->height << ", " << organized_cloud_ori->width << endl;
    
    pcl::PointCloud<PointTRGB>::Ptr mutli_pallet_cloud(new pcl::PointCloud<PointTRGB>);
    forking_pt->clear();
    for(int k = 0; k<pallet_all.size(); k++)
    {
        // cv::Mat binary_img;
        // cv::threshold(pallet_all[k].pallet_mask_ptr->image, binary_img, 0, 255, cv::THRESH_BINARY);
        // cv::imshow("solo_mask binary", binary_img);
        // cv::waitKey(1);
        
        pcl::PointCloud<PointTRGB>::Ptr one_pallet(new pcl::PointCloud<PointTRGB>);
        one_pallet->clear();

        for(int i = 0; i < cols; i++)      //x-direction
        {
            for(int j = 0; j < rows; j++)  //y-direction
            {
                int intensity = (pallet_all[k].pallet_mask_ptr->image.at<uchar>(j, i));
        
                if(intensity==255) //if(intensity>0)
                {
                    PointTRGB depth_pt = organized_cloud_ori->at(i, j);
                    if(pcl_isfinite(depth_pt.x) && pcl_isfinite(depth_pt.y) && pcl_isfinite(depth_pt.z))
                    {
                        one_pallet->push_back(depth_pt);
                    }
                }
            }      
        }          
        cout << "\t\033[1;33m#"<<k<<"\033[0m Extracted [pallet depth_cloud] pts= " << one_pallet->size() << endl;

        if(one_pallet->size()==0)
            continue;
        else
        {
            //=======================//
            // Estimate Pallet Pose
            // 1. Extract Plane; or 2. Estimate Pose (PCA)
            //=======================//
            //1. Extract Plane
            pcl::ModelCoefficients::Ptr coeff_plane(new pcl::ModelCoefficients);
            pcl::PointCloud<PointTRGB>::Ptr pallet_plane(new pcl::PointCloud<PointTRGB>);
            est_scene_plane_coeff(one_pallet, coeff_plane, pallet_plane);

            //==============================================
            // Project pallet_front onto the estimated plane
            // to reduce the noisy pallet front
            //==============================================
            for (int m=0;m<pallet_plane->points.size();m++)
            {
                PointTRGB p = pallet_plane->points[m];
                float dis = coeff_plane->values[0]*p.x+coeff_plane->values[1]*p.y+coeff_plane->values[2]*p.z+coeff_plane->values[3];
                pallet_plane->points[m].x = p.x - dis*coeff_plane->values[0];
                pallet_plane->points[m].y = p.y - dis*coeff_plane->values[1];
                pallet_plane->points[m].z = p.z - dis*coeff_plane->values[2];
            }

            //2. Estimate Pose (PCA)
            Eigen::Matrix3f vect;            
            CalculatePCA(pallet_plane, vect);

            tf2::Quaternion quat;
            quat.setRPY(0.0, 0.0, 0.0);

            tf2::Matrix3x3 mm;
            //PC1, PC2, PC3 ?? PC3, PC2, PC1 ??
            mm.setValue(vect(6),vect(3),vect(0), \
                        vect(7),vect(4),vect(1), \
                        vect(8),vect(5),vect(2));

            // Force PCA PC3 same side with camera z-axis
            if (vect(2) <0)
            {
                cout << "\033[1;33mvect(2) = \033[0m"<<vect(2) << std::endl;
                mm.setValue(vect(6),vect(3),-vect(0), \
                            vect(7),vect(4),-vect(1), \
                            vect(8),vect(5),-vect(2));
            }
            else
                cout << "\033[1;31mvect(2) = \033[0m"<<vect(2) << std::endl;

            double r, p, y;
            mm.getRPY(r,p,y);
            quat.setRPY(r,p,y);

            //============================================
            // Publish (1) Pallet Cloud Extracted from YOLOv4 BBox
            //         (2) Pallet Plane
            //============================================
            if(show_extracted_cloud==true)
            {
                pcl::toROSMsg(*one_pallet, pallet_cloud_msg);
                *mutli_pallet_cloud = *mutli_pallet_cloud + *one_pallet;
            }
            else
            {
                pcl::toROSMsg(*pallet_plane, pallet_cloud_msg);
                *mutli_pallet_cloud = *mutli_pallet_cloud + *pallet_plane;
            }
            pallet_cloud_msg.header.frame_id = "camera_color_optical_frame";
            pallet_cloud_pub.publish(pallet_cloud_msg);

            //============================================
            // Calculate & Publish Pallet Pose Arrow
            //============================================    
            // use center
            PointTRGB min_pt, max_pt, center_pt;
            pcl::getMinMax3D(*pallet_plane, min_pt, max_pt);
            center_pt.x = (min_pt.x + max_pt.x)/2.0;
            center_pt.y = (min_pt.y + max_pt.y)/2.0;
            center_pt.z = (min_pt.z + max_pt.z)/2.0;
  
            //=================
            // Forking point
            //=================
            float pallet_width = 0.250;
            float fork_len = pallet_width/4.0; //0.060;

            PointTRGB fork_pt_mid, fork_pt_left, fork_pt_right;
            fork_pt_mid.x = center_pt.x;
            fork_pt_mid.y = center_pt.y;
            fork_pt_mid.z = center_pt.z;

            fork_pt_left.x = center_pt.x + fork_len*vect(6);
            fork_pt_left.y = center_pt.y + fork_len*vect(7);
            fork_pt_left.z = center_pt.z + fork_len*vect(8);

            fork_pt_right.x = center_pt.x - fork_len*vect(6);
            fork_pt_right.y = center_pt.y - fork_len*vect(7);
            fork_pt_right.z = center_pt.z - fork_len*vect(8);

            forking_pt->push_back(fork_pt_mid);
            forking_pt->push_back(fork_pt_left);
            forking_pt->push_back(fork_pt_right);

            pcl::toROSMsg(*forking_pt, forking_pt_msg);
            forking_pt_msg.header.frame_id = "camera_color_optical_frame";
            fork_pt_pub.publish(forking_pt_msg);

            //ROS Static and Dynamic Transforms in Rviz https://www.youtube.com/watch?v=QhGxqLDeKvA
            std::string marker_coord_name = "marker_" + std::to_string(k);
            static tf2_ros::TransformBroadcaster br;
            geometry_msgs::TransformStamped transform;
            transform.header.stamp = ros::Time::now();
            transform.header.frame_id = "camera_color_optical_frame";
            transform.child_frame_id = marker_coord_name;
            transform.transform.translation.x = center_pt.x;
            transform.transform.translation.y = center_pt.y;
            transform.transform.translation.z = center_pt.z;

            transform.transform.rotation.x = quat.x();
            transform.transform.rotation.y = quat.y();
            transform.transform.rotation.z = quat.z();
            transform.transform.rotation.w = quat.w();

            br.sendTransform(transform);

            //=============================
            // Translation, Rotation Error
            //=============================
            PointTRGB ori_pt;
            ori_pt.x = 0.0;
            ori_pt.y = 0.0;
            ori_pt.z = 0.0;

            Eigen::Vector3f ori_vect = Eigen::Vector3f(0.0, 0.0, 1.0);

            float trans_err_x = center_pt.x - ori_pt.x;
            float trans_err_y = center_pt.y - ori_pt.y;
            float trans_err_z = center_pt.z - ori_pt.z;

            float inner_prod = ori_vect[0]*vect(6) + ori_vect[1]*vect(7) + ori_vect[2]*vect(8);
            float ori_vect_length = sqrt(pow(ori_vect[0], 2)+ pow(ori_vect[1], 2)+ pow(ori_vect[2], 2));
            float vect_length = sqrt(pow(vect(6), 2)+ pow(vect(7), 2)+ pow(vect(8), 2));
            float rot_err = cos(inner_prod/(ori_vect_length*vect_length));

            cout << "Translation (x, y, z), Rotation (theta) Error:\n" \
                 << trans_err_x << ", " << trans_err_y << ", " << trans_err_z << " (m)" << endl \
                 << rot_err * (180.0/3.141592653589793238463) << " (deg)"<< endl;

        }
    }

    pcl::toROSMsg(*mutli_pallet_cloud, multi_pallet_cloud_msg);
    multi_pallet_cloud_msg.header.frame_id = "camera_color_optical_frame";
    multi_pallet_cloud_pub.publish(multi_pallet_cloud_msg);
    
    return true;
}

void pallet_mask_cb(const pallet_seg::pallet_mask & pallet_mask_msg)
{
    cout << "pallet_mask_cb" << endl;
    pallet_all.clear();
    try
    {
        int pallet_num = pallet_mask_msg.masks.size();
        cout << "pallet_num size:"<< pallet_num <<endl;

        pallet_all.resize(pallet_num);

        for(int k = 0; k < pallet_num; ++k)
        {
            pallet_all[k].pallet_mask_ptr = cv_bridge::toCvCopy(pallet_mask_msg.masks[k], sensor_msgs::image_encodings::TYPE_8UC1);
        }
    }
    catch(const std::exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    bool multi_pallet_get = get_multi_pallet_cloud();
}

void organized_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& organized_cloud_msg)
{
    //==================================================//
    // Organized Point Cloud; Depth Point Cloud
    // Subscribe "/camera/depth_registered/points" topic
    //==================================================//
    cout<<"organized_cloud_cb"<<endl;
    cout<< organized_cloud_msg->header.stamp <<endl;
    int points = organized_cloud_msg->height * organized_cloud_msg->width;

    if(points!=0)
    {
        // 將點雲格式由sensor_msgs/PointCloud2轉成pcl/PointCloud(PointXYZ, PointXYZRGB)
        organized_cloud_ori->clear();
        pcl::fromROSMsg(*organized_cloud_msg, *organized_cloud_ori);

        if(input_cloud_from_file == true)
        {
            pcl::io::savePCDFileBinary<PointTRGB>(file_path_cloud_depth, *organized_cloud_ori);
            cout << "organized_cloud_ori saved: " << file_path_cloud_depth << "; (width, height) = " << organized_cloud_ori->width << ", " << organized_cloud_ori->height << endl;
        }
    }
}

int main(int argc, char** argv)
{   
    ros::init(argc, argv, "pallet_seg_pose");
    ros::NodeHandle nh;
    
    // Subscriber 
    // Synchronize https://blog.csdn.net/zhngyue123/article/details/108004007
    ros::Subscriber sub_depth_cloud = nh.subscribe("/camera/depth_registered/points", 1, organized_cloud_cb);   //depth point cloud (organized)
    ros::Subscriber sub_pallet_mask = nh.subscribe("/pallet_mask", 1, pallet_mask_cb);                          //SOLOv2 segmentation result

    // Publisher
    pallet_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> ("/pallet_cloud_pub", 1);
    multi_pallet_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> ("/multi_pallet_cloud_pub", 1);
    fork_pt_pub = nh.advertise<sensor_msgs::PointCloud2> ("/fork_pt_pub", 1);
    pallet_pose_pub = nh.advertise<visualization_msgs::Marker>("/pallet_pose", 1);

    ros::spin();

    return 0;    
}