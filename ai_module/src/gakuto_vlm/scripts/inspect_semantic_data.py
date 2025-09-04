#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge

class SemanticDataInspector:
    def __init__(self):
        rospy.init_node('semantic_data_inspector')
        self.bridge = CvBridge()
        
    def inspect_semantic_pointcloud(self):
        """セマンティック点群の詳細解析"""
        try:
            print("Waiting for /semantic_scan...")
            cloud_msg = rospy.wait_for_message("/semantic_scan", PointCloud2, timeout=10.0)
            
            print("\n" + "="*50)
            print("SEMANTIC POINT CLOUD ANALYSIS")
            print("="*50)
            
            # フィールド情報
            print("\n📊 Fields Information:")
            for i, field in enumerate(cloud_msg.fields):
                datatype_names = {1: 'INT8', 2: 'UINT8', 3: 'INT16', 4: 'UINT16', 
                                5: 'INT32', 6: 'UINT32', 7: 'FLOAT32', 8: 'FLOAT64'}
                dtype_name = datatype_names.get(field.datatype, f'UNKNOWN({field.datatype})')
                print(f"  [{i}] {field.name:15} | Offset: {field.offset:2} | Type: {dtype_name:8} | Count: {field.count}")
            
            # 基本情報
            total_points = cloud_msg.width * cloud_msg.height
            print(f"\n📐 Cloud Dimensions:")
            print(f"  Width x Height: {cloud_msg.width} x {cloud_msg.height}")
            print(f"  Total points: {total_points:,}")
            print(f"  Point step: {cloud_msg.point_step} bytes")
            print(f"  Data size: {len(cloud_msg.data):,} bytes")
            
            # データ統計
            if total_points > 0:
                print(f"\n📈 Data Statistics:")
                self._analyze_point_data(cloud_msg)
                
        except rospy.ROSException as e:
            print(f"❌ Error accessing /semantic_scan: {e}")
            
    def inspect_semantic_image(self):
        """セマンティック画像の解析"""
        try:
            print("Waiting for /camera/semantic_image...")
            img_msg = rospy.wait_for_message("/camera/semantic_image", Image, timeout=10.0)
            
            print("\n" + "="*50)
            print("SEMANTIC IMAGE ANALYSIS") 
            print("="*50)
            
            print(f"\n🖼️  Image Properties:")
            print(f"  Dimensions: {img_msg.width} x {img_msg.height}")
            print(f"  Encoding: {img_msg.encoding}")
            print(f"  Step: {img_msg.step}")
            print(f"  Data size: {len(img_msg.data):,} bytes")
            
            # OpenCV画像に変換して解析
            cv_image = self.bridge.imgmsg_to_cv2(img_msg)
            print(f"  OpenCV shape: {cv_image.shape}")
            print(f"  Data type: {cv_image.dtype}")
            
            # セマンティックIDの統計
            unique_values = np.unique(cv_image)
            print(f"\n📊 Semantic IDs found: {len(unique_values)}")
            print(f"  Range: {unique_values.min()} to {unique_values.max()}")
            print(f"  Unique values: {unique_values[:20]}{'...' if len(unique_values) > 20 else ''}")
            
        except rospy.ROSException as e:
            print(f"❌ Error accessing /camera/semantic_image: {e}")
            
    def _analyze_point_data(self, cloud_msg):
        """点群データの詳細分析"""
        field_names = [field.name for field in cloud_msg.fields]
        
        # サンプル点を取得
        sample_points = []
        sample_size = min(1000, cloud_msg.width * cloud_msg.height)
        
        count = 0
        for point in pc2.read_points(cloud_msg, field_names=field_names):
            if count >= sample_size:
                break
            sample_points.append(point)
            count += 1
            
        if sample_points:
            sample_array = np.array(sample_points)
            
            print(f"  Sample size: {len(sample_points)}")
            
            # 各フィールドの統計
            for i, field_name in enumerate(field_names):
                if i < sample_array.shape[1]:
                    values = sample_array[:, i]
                    if np.issubdtype(values.dtype, np.number):
                        print(f"  {field_name:15}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
                    else:
                        unique_vals = np.unique(values)
                        print(f"  {field_name:15}: {len(unique_vals)} unique values")
            
            # 特別なフィールドの詳細分析
            self._analyze_special_fields(sample_array, field_names)
    
    def _analyze_special_fields(self, sample_array, field_names):
        """特別なフィールド（rgb, semantic_id等）の詳細分析"""
        
        # RGB情報の分析
        if 'rgb' in field_names:
            rgb_idx = field_names.index('rgb')
            rgb_values = sample_array[:, rgb_idx].astype(np.uint32)
            
            # RGBを分解
            r = (rgb_values >> 16) & 0xFF
            g = (rgb_values >> 8) & 0xFF  
            b = rgb_values & 0xFF
            
            print(f"\n🎨 RGB Analysis:")
            print(f"  R: min={r.min()}, max={r.max()}, mean={r.mean():.1f}")
            print(f"  G: min={g.min()}, max={g.max()}, mean={g.mean():.1f}")
            print(f"  B: min={b.min()}, max={b.max()}, mean={b.mean():.1f}")
            
        # セマンティックID分析
        semantic_fields = [f for f in field_names if 'semantic' in f.lower() or 'label' in f.lower() or 'class' in f.lower()]
        for field in semantic_fields:
            if field in field_names:
                idx = field_names.index(field)
                values = sample_array[:, idx]
                unique_vals = np.unique(values)
                
                print(f"\n🏷️  {field} Analysis:")
                print(f"  Unique IDs: {len(unique_vals)}")
                print(f"  ID range: {unique_vals.min()} to {unique_vals.max()}")
                print(f"  Most frequent IDs: {unique_vals[:10]}")
    
    def run_full_inspection(self):
        """完全な検査を実行"""
        print("🔍 Starting Semantic Data Inspection...")
        print("Make sure the system is running!")
        
        self.inspect_semantic_pointcloud()
        self.inspect_semantic_image()
        
        print(f"\n✅ Inspection completed!")
        print(f"\n💡 Next steps:")
        print(f"  1. Check if rgb field contains color information")
        print(f"  2. Identify semantic_id or label field for object classes")
        print(f"  3. Map semantic IDs to object class names")

if __name__ == '__main__':
    try:
        inspector = SemanticDataInspector()
        inspector.run_full_inspection()
    except KeyboardInterrupt:
        print("\n👋 Inspection cancelled by user")