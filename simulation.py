import pygame
import numpy as np
import random
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate
import math
from shapely.geometry import LineString, Point, MultiLineString
from scipy.spatial import KDTree
from parameters import *
import threading

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
LIGHT_GRAY = (211, 211, 211)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
FPS = 60

class IncrementalKDTree:
    def __init__(self):
        # Initialize with an empty list or provided points
        self.points = []
        self.parents = {}
        # Build KDTree if there are any points
        self.tree = None
        self.path_found = False
    def insert(self, new_point, parent_point = None):
        # Append new point to the list of points
        new_point = tuple(new_point)
        self.points.append(new_point)
        # Rebuild KDTree with the new set of points
        if parent_point is not None:
            parent_point = tuple(parent_point)
            if parent_point not in self.points:
                raise Exception("Error") 
            
            self.parents[new_point] = parent_point  # Store the parent relationship

        self.tree = KDTree(self.points)
    
    def points_in_tree(self):
        return self.points
    
    def query(self, point, k=1):
        if len(self.points) == 1:
            # If only one point is in the tree, return it directly
            return self.points[0], np.linalg.norm(np.array(self.points[0]) - np.array(point))
        
        if self.tree:
            # Query the KDTree for the nearest neighbor
            distance, index = self.tree.query(point)
            nearest_point = self.points[index]
            return nearest_point, distance
        
        return None, float('inf')
    
    def get_final_path(self, start, end):
        if self.path_found:
            path = [end]
            current_point = tuple(end)
            while current_point != tuple(start):
                current_point = self.parents[current_point]
                path.append(current_point)

            path.reverse()
            return path
        
        else:
            return []

class Car:
    def __init__(self, x, y, kp_pos = 0.1, kd_pos = 0.01, kp_angular = 0.01, kd_angular = 0.01):
           self.x = np.array([
                                [x],
                                [y],
                                [0]
                            ])

           self.x_dot = np.array([
                                [0],
                                [0],
                                [0]
                            ])

           self.wheel_speed = np.array([
                                            [0],
                                            [0]
                                        ])

           self.b = 25
           self.r = 5

           self.car_dims = np.array([
                                            [-self.b, -self.b, 1],
                                            [0 		, -self.b, 1],
                                            [ self.b,  		0, 1],
                                            [ 0, 	   self.b, 1],
                                            [ -self.b, self.b, 1]
                                        ])

           self.get_transformed_pts()
           self.position = self.x[0:2].flatten()
           self.angle = self.x[2][0]
           self.previous_error = None # vector indicating error
           self.previous_pos_error = None
           self.previous_angle_error = None
           self.kp_pos = kp_pos
           self.kd_pos = kd_pos
           self.kp_angular = kp_angular
           self.kd_angular = kd_angular



    def set_wheel_velocity(self, lw_speed, rw_speed):
            self.wheel_speed = np.array([
                                            [rw_speed],
                                            [lw_speed]
                                        ])
            self.x_dot = self.forward_kinematics()

    def set_robot_velocity(self, linear_velocity, angular_velocity):
            self.x_dot = np.array([
                                            [linear_velocity],
                                            [0],
                                            [angular_velocity]
                                        ])
            self.wheel_speed = self.inverse_kinematics()


    def update_state(self, dt):
            A = np.array([
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]
                        ])
            B = np.array([
                            [np.cos(self.x[2, 0])*dt,  0],
                            [np.sin(self.x[2, 0])*dt,  0],
                            [0					 , dt]
                        ])

            vel = np.array([
                                [self.x_dot[0, 0]],
                                [self.x_dot[2, 0]]
                            ])
            self.x = A@self.x + B@vel


    def update(self, dt):
            self.wheel_speed[self.wheel_speed>MAX_WHEEL_ROT_SPEED_RAD] = MAX_WHEEL_ROT_SPEED_RAD
            self.wheel_speed[self.wheel_speed<MIN_WHEEL_ROT_SPEED_RAD] = MIN_WHEEL_ROT_SPEED_RAD
            self.x_dot = self.forward_kinematics()
            self.update_state(dt)
            self.wheel_speed = self.inverse_kinematics()
            self.position = self.x[0:2].flatten()
            self.angle = self.x[2,0]


    def get_state(self):
            return self.x, self.x_dot

    def forward_kinematics(self):
            kine_mat = np.array([
                                [self.r/2  		  , self.r/2],
                                [0 		 		  ,	0],
                                [self.r/(2*self.b), -self.r/(2*self.b)]
                                ])

            return kine_mat@self.wheel_speed

    def inverse_kinematics(self):
            ikine_mat = np.array([
                                [1/self.r, 0, self.b/self.r],
                                [1/self.r, 0, -self.b/self.r]
                                ])

            return ikine_mat@self.x_dot

    def get_transformed_pts(self):
            rot_mat = np.array([
                                [ np.cos(self.x[2, 0]), np.sin(self.x[2, 0]), self.x[0, 0]],
                                [-np.sin(self.x[2, 0]), np.cos(self.x[2, 0]), self.x[1, 0]],
                                [0, 0, 1]
                                ])

            self.car_points = self.car_dims@rot_mat.T

            self.car_points = self.car_points.astype("int")

    def get_points(self):
            self.get_transformed_pts()
            return self.car_points        
    
    def get_fov_triangle(self, fov_angle=60, fov_length=150):
        half_fov = np.radians(fov_angle / 2)
        left_vertex = self.position + fov_length * np.array([np.cos(self.angle + half_fov), np.sin(self.angle + half_fov)])
        right_vertex = self.position + fov_length * np.array([np.cos(self.angle - half_fov), np.sin(self.angle - half_fov)])
        return [self.position, left_vertex, right_vertex]

    
class DifferentialDriveBot:
    def __init__(self, init_position, init_velocity, init_angle, max_speed, max_acceleration, max_angular_rate, dt=0.1, kp_pos = 0.1, kp_angular = 0.1, kd_pos = 0.01, kd_angular = 0.01):
        self.position = np.array(init_position)
        self.velocity = np.array(init_velocity)
        self.angle = init_angle
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_angular_rate = max_angular_rate
        self.dt = dt
        self.previous_error = None # vector indicating error
        self.previous_pos_error = None
        self.previous_angle_error = None
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular

    def update(self, acceleration_input, angular_velocity):
        if acceleration_input > self.max_acceleration:
            acceleration_input = self.max_acceleration
        if acceleration_input < -self.max_acceleration:
            acceleration_input = -self.max_acceleration

        if angular_velocity > self.max_angular_rate:
            angular_velocity = self.max_angular_rate
        if angular_velocity < -self.max_angular_rate:
            angular_velocity = -self.max_angular_rate


        self.velocity += acceleration_input * self.dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        self.position += acceleration_input*np.array([math.cos(self.angle), math.sin(self.angle)])
        self.angle += angular_velocity * self.dt

    def get_fov_triangle(self, fov_angle=60, fov_length=150):
        half_fov = np.radians(fov_angle / 2)
        left_vertex = self.position + fov_length * np.array([np.cos(self.angle + half_fov), np.sin(self.angle + half_fov)])
        right_vertex = self.position + fov_length * np.array([np.cos(self.angle - half_fov), np.sin(self.angle - half_fov)])
        return [self.position, left_vertex, right_vertex]

def generate_random_polygon(center, num_sides=5, radius=50):
    angle = 2 * 3.141592653589793 / num_sides
    points = []
    for i in range(num_sides):
        offset_angle = random.uniform(-angle/4, angle/4)
        theta = i * angle + offset_angle
        x = center[0] + radius * random.uniform(0.8, 1.2) * np.cos(theta)
        y = center[1] + radius * random.uniform(0.8, 1.2) * np.sin(theta)
        points.append((x, y))
    return ShapelyPolygon(points)

def create_world(num_polygons=10, plane_size=SCREEN_WIDTH, inflation_width = 15):
    polygons = []
    inflated_polygons = []
    for _ in range(num_polygons):
        center = (random.uniform(0, plane_size), random.uniform(0, plane_size))
        num_sides = random.randint(3, 8)
        radius = random.uniform(20, 50)
        polygon = generate_random_polygon(center, num_sides, radius)
        polygons.append(polygon)

        inflated_polygon = polygon.buffer(inflation_width)
        inflated_polygons.append(inflated_polygon)

    return polygons, inflated_polygons

def draw_polygon(polygon, surface, color, scale_factor=(1, 1)):
    if isinstance(polygon, ShapelyPolygon):
        scaled_coords = [(int(x * scale_factor[0]), int(y * scale_factor[1])) for x, y in polygon.exterior.coords]
        pygame.draw.polygon(surface, color, scaled_coords)
    elif isinstance(polygon, list):  # Handles the case where the polygon is a list of points (FOV)
        scaled_coords = [(int(x * scale_factor[0]), int(y * scale_factor[1])) for x, y in polygon]
        pygame.draw.polygon(surface, color, scaled_coords)


def draw_arrow(screen, position, angle):
    # Define the arrow as a triangle with three points
    arrow_length = 20
    arrow_width = 10
    point1 = tuple(position + arrow_length * np.array([np.cos(angle), np.sin(angle)]))
    point2 = tuple(position + arrow_width * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)]))
    point3 = tuple(position + arrow_width * np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)]))
    pygame.draw.polygon(screen, RED, [point1, point2, point3])
    return ShapelyPolygon([point1, point2, point3])


def point_valid(parent, child, polygons): # returns whether a point randomly choosen is valid or not
    line = LineString([tuple(parent), tuple(child)])
    c = Point(tuple(child))
    for polygon in polygons:
        if polygon.contains(c) or line.intersects(polygon):
            return False
    
    return True

def rrt_planner(start, end, max_iter, max_distance, end_threshold, SCREEN_WIDTH, SCREEN_HEIGHT, polygons):
    i = 0
    is_reached = False
    rrt_tree = IncrementalKDTree()
    rrt_tree.insert(start)
    
    while i < max_iter:
        if random.random() < 0.1:
            temp_sample = end  # Bias towards the goal
        else:
            temp_sample = [float(random.randint(0, SCREEN_WIDTH)), float(random.randint(0, SCREEN_HEIGHT))]

        temp_parent, temp_distance = rrt_tree.query(temp_sample, k=1)

        if not point_valid(temp_parent, temp_sample, polygons):
            continue

        # Handle the case where the sample is farther than max_distance
        if temp_distance > max_distance:
            line = LineString([temp_parent, temp_sample])
            perm_sample = line.interpolate(max_distance)
            perm_sample = (perm_sample.x, perm_sample.y)  # Ensure tuple format
        else:
            perm_sample = tuple(temp_sample)

        rrt_tree.insert(perm_sample, temp_parent)

        # Check distance to the goal
        distance = math.sqrt((end[0] - perm_sample[0])**2 + (end[1] - perm_sample[1])**2)
        if distance < end_threshold:
            rrt_tree.insert(end, perm_sample)  # Properly connect the end node
            print("Path found.")
            rrt_tree.path_found = True
            return rrt_tree
        
        i += 1
    
    print("No path found.")
    rrt_tree.path_found = False  # Indicate no path was found
    return rrt_tree

def get_distance(x1, y1, x2, y2):
	return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_angle(x1, y1, x2, y2):
	# return np.arctan2(y2 - y1, x2 - x1)
	return np.arctan2(y2 - y1, x2 - x1)

def simulate_lidar(r, bot, polygons, fov_shape, fov_angle, main_screen = None):
    lidar_points = []
    for k in range(-int(r/2), int(r/2+1)):
                angle = bot.angle + k * np.radians(fov_angle)/r
                line_end = bot.position + 150 * np.array([np.cos(angle), np.sin(angle)])
                line = LineString([tuple(bot.position), tuple(line_end)])

                closest_intersection = None
                min_distance = float('inf')

                for polygon in polygons:
                    if fov_shape.intersects(polygon):
                        intersecting_area = fov_shape.intersection(polygon)
                        if not intersecting_area.is_empty:
                            # Find intersection between the line and the intersecting area
                            intersection = intersecting_area.intersection(line)
                            if not intersection.is_empty:
                                if isinstance(intersection, Point):
                                    distance = np.linalg.norm(np.array(intersection.coords[0]) - bot.position)
                                    if distance < min_distance:
                                        closest_intersection = intersection
                                        min_distance = distance
                                elif isinstance(intersection, LineString):
                                    for point in intersection.coords:
                                        point_geom = Point(point)
                                        distance = np.linalg.norm(np.array(point) - bot.position)
                                        if distance < min_distance:
                                            closest_intersection = point_geom
                                            min_distance = distance
                                elif isinstance(intersection, MultiLineString):
                                    for linestring in intersection.geoms:  # Correct way to iterate over LineStrings in a MultiLineString
                                        for point in linestring.coords:
                                            point_geom = Point(point)
                                            distance = np.linalg.norm(np.array(point) - bot.position)
                                            if distance < min_distance:
                                                closest_intersection = point_geom
                                                min_distance = distance

                # Transform the closest intersection point to the bot's frame
                if closest_intersection:
                    lidar_points.append([closest_intersection.x, closest_intersection.y])
                    if main_screen:
                        pygame.draw.circle(main_screen,RED, (closest_intersection.x, closest_intersection.y), 2)
    
    return lidar_points

def PID_path_follower(bot, desired_position, dt = 0.1, ang_setpoint = None):
    current_pos = bot.position
    current_angle = bot.angle
    error = desired_position - current_pos
    error_pos = np.linalg.norm(error)
    if ang_setpoint:
        desired_angle = ang_setpoint
    else:
        desired_angle = math.atan2(desired_position[1] - bot.position[1], desired_position[0] - bot.position[0])
    error_ang = desired_angle - current_angle
    error_ang = (error_ang + np.pi) % (2 * np.pi) - np.pi

    # error_ang = (error_ang + np.pi) % (2 * np.pi) - np.pi  # This ensures it's between -π and π
    acc_p = bot.kp_pos*error_pos
    ang_p = bot.kp_angular*error_ang

    if not bot.previous_pos_error or not bot.previous_ang_error:
         bot.previous_pos_error = error_pos
         bot.previous_ang_error = error_ang
    else:
         bot.previous_pos_error = error_pos
         bot.previous_ang_error = error_ang
         acc_p += bot.kd_pos*(error_pos - bot.previous_pos_error)/dt
         ang_p += bot.kd_angular*(error_ang - bot.previous_ang_error)/dt

    ang_p = np.clip(ang_p, -4, 4)
    acc_p = np.clip(acc_p, -10, 10)


    return acc_p , ang_p

def waypoint_generator(path, bot, current_index): # gives the index of waypoint that needs to be taken as goal
    if current_index == len(path)-1:
         return None
    if current_index == None:
         return -1
    if np.linalg.norm(bot.position - path[current_index]) < 15:
         return current_index+1
    else:
         return current_index
    

def simulate_motion():
    pygame.init()
    main_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Bot Simulation")
    polygon_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    polygon_surface.fill(WHITE)  # Fill with the background color
        
    boundary_color = BLACK  # Color for the boundary around the FOV screen
    boundary_thickness = 2  # Thickness of the boundary
    clock = pygame.time.Clock()
    polygons, inflated_obs = create_world(num_polygons = 80)
    polygons_draw = True
    lidar = True

    if polygons_draw:
        for polygon in polygons:
            draw_polygon(polygon, polygon_surface, BLACK)

    start_pos = []
    end_pos = []
    path_following = True
    realistic = True

    while True:
        initial_position = [float(random.randint(0, SCREEN_WIDTH)), float(random.randint(0, SCREEN_HEIGHT))]
        if not realistic:
            bot = DifferentialDriveBot(init_position=initial_position, init_velocity=[0.0, 0.0],
                                   init_angle=0, max_speed=2, max_acceleration=2, max_angular_rate = 1, kp_pos=0.05, kp_angular=0.01)
        else:
            bot = Car(initial_position[0], initial_position[1], kp_pos = 1, kp_angular = 0.6, kd_angular=0.001)
        bot_shape = draw_arrow(main_screen, bot.position, bot.angle)
        if not any(bot_shape.intersects(polygon) for polygon in polygons):
            start_pos = initial_position
            break 
    
    # Generate end point ensuring it's in free space
    while True:
        end = [float(random.randint(0, SCREEN_WIDTH)), float(random.randint(0, SCREEN_HEIGHT))]
        end_point = Point(tuple(end))
        
        # Check if the end point is in free space (not inside any polygon)
        if not any(polygon.contains(end_point) for polygon in polygons):
            end_pos = end
            break  # Exit the loop when a valid end point is found
    
    acceleration_input = 0.0
    angular_velocity = 0.0
    goal_index = 0
    rrt_tree = rrt_planner(start_pos, end_pos, 1000, 50, 50, SCREEN_WIDTH, SCREEN_HEIGHT, inflated_obs)
    path_rrt = rrt_tree.get_final_path(start_pos, end_pos)
    running = True
    while running:
        main_screen.blit(polygon_surface, (0, 0))

        pygame.draw.circle(main_screen,GREEN, (start_pos[0], start_pos[1]), 10)
        pygame.draw.circle(main_screen,BLUE, (end_pos[0], end_pos[1]), 10)
        pygame.draw.circle(main_screen,BLUE, (end_pos[0], end_pos[1]), 50, width=1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if not realistic:
            if keys[pygame.K_UP]:
                acceleration_input += 0.25
            if keys[pygame.K_DOWN]:
                acceleration_input -= 0.25
            if keys[pygame.K_LEFT]:
                angular_velocity -= 0.05
            if keys[pygame.K_RIGHT]:
                angular_velocity += 0.05
            if keys[pygame.K_c]:
                angular_velocity = 0.0
            if keys[pygame.K_b]:
                acceleration_input = 0.0
            if keys[pygame.K_ESCAPE]:
                running = False
            
            if path_following and path_rrt != []:
                goal_index = waypoint_generator(path_rrt, bot, goal_index)
                print("current goal: ", path_rrt[goal_index])
                if goal_index:
                    acc_des, ang_des = PID_path_follower(bot, path_rrt[goal_index])
                    acceleration_input += acc_des
                    angular_velocity += ang_des
            
            bot.update(acceleration_input, angular_velocity)
        else:
            if keys[pygame.K_UP]:
                acceleration_input += 0.25
            if keys[pygame.K_DOWN]:
                acceleration_input -= 0.25
            if keys[pygame.K_LEFT]:
                angular_velocity -= 0.05
            if keys[pygame.K_RIGHT]:
                angular_velocity += 0.05
            if keys[pygame.K_c]:
                angular_velocity = 0.0
            if keys[pygame.K_b]:
                acceleration_input = 0.0
            if keys[pygame.K_ESCAPE]:
                running = False
            
            if path_following and path_rrt != []:
                goal_index = waypoint_generator(path_rrt, bot, goal_index)
                if np.linalg.norm(bot.position - end_pos) < 40:
                    path_following = False
                    acceleration_input = 0
                    angular_velocity = 0
                    goal_index = None
                if goal_index:
                    acc_des, ang_des = PID_path_follower(bot, path_rrt[goal_index])
                    acceleration_input = acc_des
                    angular_velocity = ang_des
                    

            bot.set_robot_velocity(acceleration_input, angular_velocity)
            bot.update(dt = 0.1)



        collision_detected = False
        
        bot_shape = draw_arrow(main_screen, bot.position, bot.angle)


        for polygon in polygons:
            if bot_shape.intersects(polygon):
                acceleration_input = 0.0
                angular_velocity = 0.0
                collision_detected = True
                break

        # Handle collision
        if collision_detected:
            print("Collision detected!")
            #acceleration_input = 0.0
            #angular_velocity = 0.0

        fov_angle = 60
        fov_triangle = bot.get_fov_triangle(fov_angle=60)
        pygame.draw.polygon(main_screen, GREEN, fov_triangle, 2)
        
        fov_shape = ShapelyPolygon(fov_triangle)
        r = 10
        
        if rrt_tree.path_found:
            for i in range(0, len(path_rrt)-2):
                pygame.draw.line(main_screen, RED, path_rrt[i], path_rrt[i+1])
        else:
            for child, parent in rrt_tree.parents.items():
                pygame.draw.line(main_screen, RED, parent, child)

        lidar_points = []
        if lidar:
            lidar_points = simulate_lidar(r, bot, polygons, fov_shape, fov_angle, main_screen)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    simulate_motion()
