import os
import numpy as np

MAP_DICT = {
                0: 'Austin',
                1: 'BrandsHatch',
                2: 'Budapest', 
                3: 'Catalunya', 
                4: 'Hockenheim', 
                5: 'IMS', 
                6: 'Melbourne', 
                7: 'MexicoCity', 
                8: 'Monza', 
                9: 'MoscowRaceway', 
                10: 'Nuerburgring', 
                11: 'Oschersleben', 
                12: 'Sakhir', 
                13: 'SaoPaulo', 
                14: 'Sepang', 
                15: 'Silverstone', 
                16: 'Sochi', 
                17: 'Spa', 
                18: 'Spielberg', 
                19: 'YasMarina', 
                20: 'Zandvoort'
}

class MapManager:
    def __init__(
        self,
        map_name: str,
        map_dir: str = os.path.dirname(__file__),
        map_ext: str = '.png',
        speed: float = 5.0,
        downsample: int = 1
    ):
        # 基本設定をメンバに保持
        self.map_dir     = map_dir
        self.map_ext     = map_ext
        self.speed       = speed
        self.downsample  = downsample

        # 初回ロード
        self._set_map_name(map_name)
        self._load_map_data()

    def _set_map_name(self, map_name: str):
        """マップ名から各種パスを再設定"""
        self.map_name      = map_name
        self.map_base_dir  = os.path.join(self.map_dir, map_name)
        self.map_path      = os.path.join(self.map_base_dir, map_name + "_map")
        self.map_img_path  = os.path.join(self.map_base_dir, map_name + self.map_ext)
        self.map_yaml_path = os.path.join(self.map_base_dir, map_name + '_map.yaml')
        self.center_line_path = os.path.join(self.map_base_dir, map_name + '_centerline.csv')
        self.race_line_path   = os.path.join(self.map_base_dir, map_name + '_raceline.csv')

    def _load_map_data(self):
        """ウェイポイント読み込み→ダウンサンプル→速度付与→累積距離計算"""
        # CSV から中心線ウェイポイントを読み込み
        wpts = np.genfromtxt(self.center_line_path, delimiter=',', usecols=(0, 1))
        # ダウンサンプル
        wpts = wpts[::self.downsample]
        # 速度列を付与
        speeds = np.full((wpts.shape[0], 1), self.speed, dtype=np.float32)
        self.waypoints = np.column_stack((wpts, speeds))  # shape=(N,3)

        # 各セグメント長・累積距離を計算
        diffs = np.diff(self.waypoints[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.cum_dis    = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        self.total_dis  = float(self.cum_dis[-1])

    def update_map(
        self,
        new_map_name: str,
        speed: float = None,
        downsample: int = None
    ):
        """
        マップ名／速度／ダウンサンプル率を更新して、ウェイポイントや距離データを再読み込みします。

        :param new_map_name: 新しいマップ名（MAP_DICT のキーまたは文字列）
        :param speed:       （省略可）速度を新しく設定する場合に指定
        :param downsample:  （省略可）ダウンサンプル率を新しく設定する場合に指定
        """
        # オプションのパラメータを反映
        if speed       is not None:  self.speed      = speed
        if downsample  is not None:  self.downsample = downsample

        # マップ名・パスを再設定し、データを再ロード
        self._set_map_name(new_map_name)
        self._load_map_data()
        
    def get_trackline_segment(self, point):
        wpts = self.waypoints[:, :2]
        # Convert list of tuples to numpy array for efficient calculations
        wpts_array = np.array(wpts)
        point_array = np.array(point).reshape(1, 2)  # pointを2次元配列に変換して形状を合わせる
    
        # Calculate the distance from the point to each of the waypoints
        dists = np.linalg.norm(point_array - wpts_array, axis=1)
    
        # Find the segment that is closest to the point
        min_dist_segment = np.argmin(dists)
        if min_dist_segment == 0:
            return 0, dists
        elif min_dist_segment == len(dists)-1:
            return len(dists)-2, dists 

        if dists[min_dist_segment+1] < dists[min_dist_segment-1]:
            return min_dist_segment, dists
        else: 
            return min_dist_segment - 1, dists
        
    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        d_ss = self.cum_dis[idx+1] - self.cum_dis[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            Area = Area_square**0.5
            h = Area * 2/d_ss
            if np.isnan(h):
                h = 0
            x = (d1**2 - h**2)**0.5

        return x, h
    
    def get_future_waypoints(self, current_point, num_points=10):
        idx, _ = self.get_trackline_segment(current_point)
        
        # インデックスのリストを作成（周回を考慮）
        future_indices = [(idx + i) % len(self.waypoints) for i in range(num_points)]
        
        # インデックスに対応するウェイポイントを取得
        future_wpts = self.waypoints[future_indices]
        
        return future_wpts




    def calc_progress(self, point):
        idx, dists = self.get_trackline_segment(point)

        x, h = self.interp_pts(idx, dists)

        s = self.cum_dis[idx] + x

        s = s/self.total_dis * 100
        
        return s
    