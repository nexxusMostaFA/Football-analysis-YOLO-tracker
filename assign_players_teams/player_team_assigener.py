from sklearn.cluster import KMeans
import cv2

class PlayerTeamAssigner():

    def __init__(self):
       self.teams_dict = {}
       self.team_colors_assigner = {}

    def cluster_model(self , image):
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)        
        kmeans.fit(image)
        return kmeans
    
    def get_tshirt_color(self , frame ,bbox ):
        image = frame[int(bbox[1]) : int(bbox[3])  ,int(bbox[0]) :int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2)  , : ]
        image_2d = top_half_image.reshape(-1,3)
        kmeans  = self.cluster_model(image_2d)
        labels = kmeans.labels_
        clustured_img = labels.reshape(top_half_image.shape[0] , top_half_image.shape[1])
        corner_points = [clustured_img[0,0] ,clustured_img[0,-1] ,clustured_img[-1,0] , clustured_img[-1,-1]]
        background = max(set(corner_points) , key=corner_points.count)
        player = 1- background
        player_tshirt_color = kmeans.cluster_centers_[player]
        
        return player_tshirt_color
    
    
    def assign_colors(self , frame  , player_track):
        player_colors = []
        for tarck_id , player_info in player_track.items():
            bbox = player_info['bbox']
            player_tshirt_color = self.get_tshirt_color(frame , bbox)
            player_colors.append(player_tshirt_color)

        self.kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        self.kmeans.fit(player_colors)

        self.teams_dict[1] = self.kmeans.cluster_centers_[0]
        self.teams_dict[2] = self.kmeans.cluster_centers_[1]


    def assign_teams(self  , frame , player_bbox , player_track_id):
        if player_track_id in self.team_colors_assigner:
            return self.team_colors_assigner[player_track_id]
        
        plater_tshirt_color = self.get_tshirt_color(frame , player_bbox)
        team_id = self.kmeans.predict(plater_tshirt_color.reshape(1,-1))[0]
        team_id += 1

        self.team_colors_assigner[player_track_id] = team_id

        return team_id


        








        
        

