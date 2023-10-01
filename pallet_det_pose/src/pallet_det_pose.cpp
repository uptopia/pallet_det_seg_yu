/** @file pallet_det_pose.cpp
  * @brief 使用YOLOv4托盤邊界框內的點雲，估測托盤姿態

  * 輸入：(1)/camera/depth_registered/points RGBD相機之有序點雲
         (2)/darknet_ros/bounding_boxes YOLOv4_ros邊界框
  * 輸出：(1)/pallet_cloud_pub 邊界框內提取的點雲 
  *      (2)/pallet_pose 托盤姿態
  * depth_cloud_cb：轉換有序點雲格式
  * yolo_cb：讀取YOLOv4邊界框
  * get_pallet_info：針對每一個邊界框
  *      (1)提取邊界框內點雲 (2)est_scene_plane_coeff估測姿態 (3)發佈托盤點雲&姿態
  * est_scene_plane_coeff：以SAC_RANSAC擬合邊界框提取出的托盤點雲，
  *      擬合之平面法向量即為托盤姿態
*/
// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
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

// boost
#include <boost/make_shared.hpp>

// YOLO pallet_det
#include <pallet_det/bboxes.h>
#include <pallet_det/bbox.h>

//C++ 
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#define PI 3.1415926

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
    std::string pallet_class;
    float probability;
    Box2D box_pixel;
    Center2D center_pixel;
    Center3D center_point;
    PointCloudTRGBPtr depth_cloud;
};

//=====Parameters Setting=====//
bool input_cloud_from_file = false;
std::string file_path_cloud_depth = "./src/cloud_yolo/depth_cloud_tmp.pcd";
bool show_plane_remain = false;       //true:顯示RANSAC擬合之托盤平面＆平面外剩餘點雲; false：不顯示
bool show_extracted_cloud = false;    //true:顯示從邊界框提取之點雲; false：最終擬合到邊界框提取之點雲的平面點
//=====Parameters Setting=====//

std::vector<std::string> pallet_labels{}; //YOLOv4偵測到的托盤類別(字串)
int num_unique_pallet_label = 0;          //YOLOv4偵測到的托盤不重複的類別個數
std::vector<Pallet> pallet_all{};         //所有YOLOv4偵測到的托盤資訊(含邊界框、點雲、中心點)

int img_width;
int img_height;

pcl::PointCloud<PointTRGB>::Ptr organized_cloud_ori(new pcl::PointCloud<PointTRGB>);
pcl::PointCloud<PointTRGB>::Ptr forking_pt(new pcl::PointCloud<PointTRGB>);

ros::Publisher pallet_cloud_pub, multi_pallet_cloud_pub, pallet_pose_pub, fork_pt_pub;
sensor_msgs::PointCloud2 pallet_cloud_msg, multi_pallet_cloud_msg, forking_pt_msg;

void euler_from_quaternion(float x, float y, float z, float w, float& roll_x, float& pitch_y, float& yaw_z)
{
    // Convert a quaternion into euler angles (roll, pitch, yaw)
    // roll is rotation around x in radians (counterclockwise)
    // pitch is rotation around y in radians (counterclockwise)
    // yaw is rotation around z in radians (counterclockwise)
    // https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/

    float t0 = +2.0 * (w * x + y * z);
    float t1 = +1.0 - 2.0 * (x * x + y * y);
    roll_x = atan2(t0, t1);
    
    float t2 = +2.0 * (w * y - z * x);
    if(t2 > 1.0)
        t2 = 1.0;
    else if(t2 <-1.0)
        t2 = -1.0;    
    pitch_y = asin(t2);
    
    float t3 = +2.0 * (w * z + x * y);
    float t4 = +1.0 - 2.0 * (y * y + z * z);
    yaw_z = atan2(t3, t4);
    
    cout << "Quaternion (x, y, z, w) = " << x <<"," << y <<"," << z <<"," << w <<endl;
    cout << "Euler (roll_x, pitch_y, yaw_z) = "<< roll_x*180.0f/PI <<"," << pitch_y*180.0f/PI <<"," << yaw_z*180.0f/PI <<endl;
    // return roll_x, pitch_y, yaw_z # in radians
}

void est_scene_plane_coeff(pcl::PointCloud<PointTRGB>::Ptr pallet_cloud_ori, pcl::ModelCoefficients::Ptr coeff_plane, pcl::PointCloud<PointTRGB>::Ptr pallet_plane)
{
    pallet_plane->clear();
    pcl::PointCloud<PointTRGB>::Ptr pallet_plane_outlier(new pcl::PointCloud<PointTRGB>);
    pcl::PointIndices::Ptr inlier_plane(new pcl::PointIndices);

    pcl::SACSegmentation<PointTRGB> seg_plane;
    seg_plane.setOptimizeCoefficients(true);
    seg_plane.setModelType(pcl::SACMODEL_PLANE);
    seg_plane.setMethodType(pcl::SAC_RANSAC);
    seg_plane.setDistanceThreshold(0.010);      //0.03m內的點雲
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

bool get_pallet_info()
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
            << "*****Get Pallet Info, START!*****\n"
            << "//===============================//\n\n";
        
    //========================================//
    // Go over all EVERY YOLOv4 detected pallets
    // save 2D, 3D pallet information
    //========================================//
    int front_cnt = 0;
    for(int n = 0; n < pallet_all.size(); ++n)
    {
        if((pallet_all[n].pallet_class).compare("front")==0 && (front_cnt==0))
        {
            front_cnt += 1;
            cout << "Pallet #" << n << endl;
            //==================================//
            // Extract Pallet Cloud
            // (2D pixel mapping to 3D points)
            //==================================//
            pallet_all[n].depth_cloud = boost::make_shared<pcl::PointCloud<PointTRGB>>();

            int xmin = pallet_all[n].box_pixel.xmin;
            int xmax = pallet_all[n].box_pixel.xmax;
            int ymin = pallet_all[n].box_pixel.ymin;
            int ymax = pallet_all[n].box_pixel.ymax;
            
            cout << "\timgwidth, imgHeight = " << img_width << ", " << img_height << endl;
            cout << "\tPixel (xmin, xmax, ymin, ymax) = "
                 << xmin << ", " << xmax <<", " << ymin << ", " << ymax << endl;

            //Map 2D pixel to 3D points
            const clock_t cloud_extract_start = clock();
            for(int i = xmin; i <= xmax; i++)
            {
                for(int j = ymin; j<= ymax; j++)
                {
                    PointTRGB depth_pt = organized_cloud_ori->at(i, j);
                    if(pcl_isfinite(depth_pt.x) && pcl_isfinite(depth_pt.y) && pcl_isfinite(depth_pt.z))
                    {
                        pallet_all[n].depth_cloud->push_back(depth_pt);
                    }
                }
            }
            const clock_t cloud_extract_end = clock();
            cout << "cloud_extract_end - cloud_extract_start: " << float( cloud_extract_end - cloud_extract_start ) /  CLOCKS_PER_SEC << std::endl;
            cout << "\tExtracted [pallet depth_cloud] pts= " << pallet_all[n].depth_cloud->size() << endl;

            //=======================//
            // Estimate Pallet Pose
            //=======================//
            //1. Extract Plane
            //2. Estimate Pose (PCA)
            pcl::PointCloud<PointTRGB>::Ptr pallet_ori(new pcl::PointCloud<PointTRGB>);
            pcl::copyPointCloud(*pallet_all[n].depth_cloud, *pallet_ori);

            const clock_t object_pose_est_beg = clock();
            pcl::ModelCoefficients::Ptr coeff_plane(new pcl::ModelCoefficients);
            pcl::PointCloud<PointTRGB>::Ptr pallet_plane(new pcl::PointCloud<PointTRGB>);
            est_scene_plane_coeff(pallet_ori, coeff_plane, pallet_plane);
            
            //============================================
            // Publish (1) Pallet Cloud Extracted from YOLOv4 BBox
            //         (2) Pallet Plane
            //============================================
            if(show_extracted_cloud==true)
                pcl::toROSMsg(*pallet_ori, pallet_cloud_msg);
            else
                pcl::toROSMsg(*pallet_plane, pallet_cloud_msg);
            pallet_cloud_msg.header.frame_id = "camera_color_optical_frame";
            pallet_cloud_pub.publish(pallet_cloud_msg);
            
            //============================================
            // Calculate & Publish Pallet Pose Arrow
            //============================================
            // // use centroid
            // Eigen::Vector4f centroid;
            // pcl::compute3DCentroid(*pallet_plane, centroid);
            // cout << "Centroid:" << "(" << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ")" << endl;
            
            // use center
            PointTRGB min_pt, max_pt, center_pt;
            pcl::getMinMax3D(*pallet_plane, min_pt, max_pt);
            center_pt.x = (min_pt.x + max_pt.x)/2.0;
            center_pt.y = (min_pt.y + max_pt.y)/2.0;
            center_pt.z = (min_pt.z + max_pt.z)/2.0;

            Eigen::Vector3f vv1 = Eigen::Vector3f(1.0, 0.0, 0.0);
            Eigen::Vector3f vv2 = Eigen::Vector3f(coeff_plane->values[0], coeff_plane->values[1], coeff_plane->values[2]);
            vv2.normalize();

            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(vv1, vv2);
            const clock_t object_pose_est_end = clock();
            cout << "object_pose_est_end - object_pose_est_beg: " << float( object_pose_est_end - object_pose_est_beg ) /  CLOCKS_PER_SEC << std::endl;

            //=========rviz marker=========
            visualization_msgs::Marker pallet_arrow;
            pallet_arrow.header.frame_id = "camera_color_optical_frame";
            pallet_arrow.header.stamp = ros::Time();
            pallet_arrow.ns = "my_namespace";
            pallet_arrow.id = 0;
            pallet_arrow.type = visualization_msgs::Marker::ARROW;
            pallet_arrow.action = visualization_msgs::Marker::ADD;
            pallet_arrow.pose.position.x = center_pt.x; //centroid[0];
            pallet_arrow.pose.position.y = center_pt.y; //centroid[1];
            pallet_arrow.pose.position.z = center_pt.z; //centroid[2];
            pallet_arrow.pose.orientation.x = q.x();
            pallet_arrow.pose.orientation.y = q.y();
            pallet_arrow.pose.orientation.z = q.z();
            pallet_arrow.pose.orientation.w = q.w();
            pallet_arrow.scale.x = 0.150;
            pallet_arrow.scale.y = 0.008;  //width
            pallet_arrow.scale.z = 0.008;  //height
            pallet_arrow.color.a = 1.0;    // Don't forget to set the alpha!
            pallet_arrow.color.r = 1.0;
            pallet_arrow.color.g = 0.0;
            pallet_arrow.color.b = 0.0;

            cout<< "pallet_pose_topic rviz marker" <<endl;

            pallet_pose_pub.publish(pallet_arrow);
            //=========rviz marker=========
        }
    }

    return true;
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
    pcl::PointCloud<PointTRGB>::Ptr mutli_pallet_cloud(new pcl::PointCloud<PointTRGB>);
    forking_pt->clear();
    for(int k = 0; k<pallet_all.size(); k++)
    {
        pcl::PointCloud<PointTRGB>::Ptr one_pallet(new pcl::PointCloud<PointTRGB>);
        one_pallet->clear();

        //==================================//
        // Extract Pallet Cloud
        // (2D pixel mapping to 3D points)
        //==================================//
        int xmin = pallet_all[k].box_pixel.xmin;
        int xmax = pallet_all[k].box_pixel.xmax;
        int ymin = pallet_all[k].box_pixel.ymin;
        int ymax = pallet_all[k].box_pixel.ymax;
        
        cout << "\timgwidth, imgHeight = " << img_width << ", " << img_height << endl;
        cout << "\tPixel (xmin, xmax, ymin, ymax) = "
                << xmin << ", " << xmax <<", " << ymin << ", " << ymax << endl;

        //Map 2D pixel to 3D points
        const clock_t cloud_extract_start = clock();
        for(int i = xmin; i <= xmax; i++)
        {
            for(int j = ymin; j<= ymax; j++)
            {
                PointTRGB depth_pt = organized_cloud_ori->at(i, j);
                if(pcl_isfinite(depth_pt.x) && pcl_isfinite(depth_pt.y) && pcl_isfinite(depth_pt.z))
                {
                    one_pallet->push_back(depth_pt);
                }
            }
        }
        const clock_t cloud_extract_end = clock();
        cout << "cloud_extract_end - cloud_extract_start: " << float( cloud_extract_end - cloud_extract_start ) /  CLOCKS_PER_SEC << std::endl;
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

            //============================================
            // Forking point
            //============================================
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

            cout << "euler_from_quaternion" <<endl;
            float roll_x, pitch_y, yaw_z;
            euler_from_quaternion(quat.x(), quat.y(), quat.z(), quat.w(), roll_x, pitch_y, yaw_z);

            br.sendTransform(transform);
        }
    }

    pcl::toROSMsg(*mutli_pallet_cloud, multi_pallet_cloud_msg);
    multi_pallet_cloud_msg.header.frame_id = "camera_color_optical_frame";
    multi_pallet_cloud_pub.publish(multi_pallet_cloud_msg);
    
    return true;
}

void pallet_box_cb(const pallet_det::bboxes::ConstPtr& boxes_msg)
{
    //==================================================//
    // Subscribe "/pallet_det/bboxes" topic
    //==================================================//

    int obj_num = boxes_msg->bboxes.size();
    
    pallet_labels = {};
    pallet_all.resize(obj_num);

    int cnt = 0;
    for(int k = 0; k < obj_num; ++k)
    {
        int xmin = boxes_msg->bboxes[k].xmin;
        int xmax = boxes_msg->bboxes[k].xmax;
        int ymin = boxes_msg->bboxes[k].ymin;
        int ymax = boxes_msg->bboxes[k].ymax;
        cout << "\t===Pixel (xmin, xmax, ymin, ymax) = "
                << xmin << ", " << xmax <<", " << ymin << ", " << ymax << endl;
        std::string pallet_class = boxes_msg->bboxes[k].object_name;
        if(pallet_class.compare("front")==0)
        {
            float probability = boxes_msg->bboxes[k].score;
            int xmin = boxes_msg->bboxes[k].xmin;
            int xmax = boxes_msg->bboxes[k].xmax;
            int ymin = boxes_msg->bboxes[k].ymin;
            int ymax = boxes_msg->bboxes[k].ymax;
            int center_x = int((xmin +xmax)/2.0);
            int center_y = int((ymin +ymax)/2.0);
            
            pallet_all[cnt].pallet_class = pallet_class;
            pallet_all[cnt].probability = probability;
            pallet_all[cnt].box_pixel.xmin = xmin;
            pallet_all[cnt].box_pixel.xmax = xmax;
            pallet_all[cnt].box_pixel.ymin = ymin;
            pallet_all[cnt].box_pixel.ymax = ymax;
            pallet_all[cnt].center_pixel.x = center_x;
            pallet_all[cnt].center_pixel.y = center_y;

            // cout << "\t===Pixel (xmin, xmax, ymin, ymax) = "
            //     << xmin << ", " << xmax <<", " << ymin << ", " << ymax << endl;
            pallet_labels.push_back(pallet_class);

            cnt++;
        }
    }

    pallet_all.resize(cnt);
    //remove duplicate pallet_labels
    std::sort(pallet_labels.begin(), pallet_labels.end());
    pallet_labels.erase(std::unique(pallet_labels.begin(), pallet_labels.end()), pallet_labels.end());
    num_unique_pallet_label = pallet_labels.size();

    if(boxes_msg->bboxes.empty())
        obj_num = 0;
    if(pallet_labels.empty())
        num_unique_pallet_label = 0;

    cout << "Total YOLO clusters = " << obj_num << endl;   //ERROR: display 1 even if no obj detected
    cout << "Total num of pallet types = " << num_unique_pallet_label << endl; //ERROR: display 1 even if no obj detected

    // bool pallet_get = get_pallet_info();
    bool multi_pallet_get = get_multi_pallet_cloud();
}

void organized_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& organized_cloud_msg)
{
    //==================================================//
    // Organized Point Cloud; Depth Point Cloud
    // Subscribe "/camera/depth_registered/points" topic
    //==================================================//
    cout<<"organized_cloud_cb"<<endl;
    int points = organized_cloud_msg->height * organized_cloud_msg->width;

    if(points!=0)
    {
        img_width = organized_cloud_msg->width;
        img_height = organized_cloud_msg->height;

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
    ros::init(argc, argv, "pallet_det_pose");
    ros::NodeHandle nh;
    
    // Subscriber
    ros::Subscriber sub_depth_cloud = nh.subscribe("/camera/depth_registered/points", 1, organized_cloud_cb); //organized point cloud
    ros::Subscriber sub_yolov4 = nh.subscribe("/darknet_ros/bounding_boxes", 1, pallet_box_cb);   // yolo_cb);                    //yolov4 detection result
 
    // Publisher
    pallet_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> ("/pallet_cloud_pub", 1);
    multi_pallet_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> ("/multi_pallet_cloud_pub", 1);
    fork_pt_pub = nh.advertise<sensor_msgs::PointCloud2> ("/fork_pt_pub", 1);
    pallet_pose_pub = nh.advertise<visualization_msgs::Marker>("/pallet_pose", 1);

    ros::spin();

    return 0;    
}