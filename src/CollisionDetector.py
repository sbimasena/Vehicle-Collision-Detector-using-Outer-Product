import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Vector:
    x: float
    y: float
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)

class VehiclePathVisualizer:
    def __init__(self):
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y']
        self.current_color = 0
    
    def get_next_color(self):
        color = self.colors[self.current_color]
        self.current_color = (self.current_color + 1) % len(self.colors)
        return color
    
    def setup_plot(self, title="Vehicle Collision Detection"):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.grid(True)
        plt.axis('equal')
    
    def plot_path(self, start: Point, end: Point, label: str):
        color = self.get_next_color()
        plt.plot([start.x, end.x], [start.y, end.y], 
                color=color, linewidth=2, label=label)
        
        plt.plot(start.x, start.y, color=color, marker='o', markersize=8)
        plt.plot(end.x, end.y, color=color, marker='s', markersize=8)
    
    def plot_intersection(self, point: Point, is_valid: bool):
        if is_valid:
            plt.plot(point.x, point.y, 'k*', markersize=15, label='Valid Intersection')
        else:
            plt.plot(point.x, point.y, 'rx', markersize=10, label='Invalid Intersection')
    
    def add_legend(self):
        plt.legend()
    
    def show_plot(self):
        plt.show()

class VehicleCollisionDetector:
    def __init__(self):
        self.tolerance = 1e-10
        self.visualizer = VehiclePathVisualizer()
    
    def outer_product(self, v1: Vector, v2: Vector) -> float:
        return v1.x * v2.y - v1.y * v2.x
    
    def vector_from_points(self, start: Point, end: Point) -> Vector:
        return Vector(end.x - start.x, end.y - start.y)
    
    def point_to_vector(self, p: Point) -> Vector:
        return Vector(p.x, p.y)
    
    def vector_to_point(self, v: Vector) -> Point:
        return Point(v.x, v.y)
    
    def find_intersection(self, r: Point, a: Vector, s: Point, b: Vector) -> Optional[Point]:
        a_wedge_b = self.outer_product(a, b)
        
        if abs(a_wedge_b) < self.tolerance:
            return None
        
        r_vec = self.point_to_vector(r)
        s_vec = self.point_to_vector(s)
        
        s_wedge_b = self.outer_product(s_vec, b)
        r_wedge_a = self.outer_product(r_vec, a)
        
        coef_a = s_wedge_b / a_wedge_b
        coef_b = r_wedge_a / (-a_wedge_b)
        
        result_vector = (a * coef_a) + (b * coef_b)
        return self.vector_to_point(result_vector)
    
    def detect_collision(self, vehicle1_path: Tuple[Point, Point], 
                        vehicle2_path: Tuple[Point, Point],
                        visualize: bool = False,
                        plot_title: str = "Vehicle Collision Detection") -> Tuple[bool, Optional[Point], Optional[Point]]:
        if visualize:
            self.visualizer.setup_plot(plot_title)
            self.visualizer.plot_path(vehicle1_path[0], vehicle1_path[1], "Vehicle 1 Path")
            self.visualizer.plot_path(vehicle2_path[0], vehicle2_path[1], "Vehicle 2 Path")
        
        a = self.vector_from_points(vehicle1_path[0], vehicle1_path[1])
        b = self.vector_from_points(vehicle2_path[0], vehicle2_path[1])
        
        intersection = self.find_intersection(vehicle1_path[0], a, vehicle2_path[0], b)
        
        if intersection is None:
            if visualize:
                self.visualizer.add_legend()
                self.visualizer.show_plot()
            return False, None, None
        
        def is_point_in_segment(point: Point, segment_start: Point, segment_end: Point) -> bool:
            buffer = 1e-10
            return (min(segment_start.x, segment_end.x) - buffer <= point.x <= max(segment_start.x, segment_end.x) + buffer and
                    min(segment_start.y, segment_end.y) - buffer <= point.y <= max(segment_start.y, segment_end.y) + buffer)
        
        is_valid_intersection = (is_point_in_segment(intersection, vehicle1_path[0], vehicle1_path[1]) and
                               is_point_in_segment(intersection, vehicle2_path[0], vehicle2_path[1]))
        
        if visualize:
            if is_valid_intersection:
                self.visualizer.plot_intersection(intersection, True)
            self.visualizer.add_legend()
            self.visualizer.show_plot()
        
        return is_valid_intersection, intersection if is_valid_intersection else None, intersection


def main():
    detector = VehicleCollisionDetector()
    
    # To simulate other test cases, simply change the path below
    vehicle1_path = (Point(0, 0), Point(4, 4))
    vehicle2_path = (Point(0, 4), Point(4, 0))
    
    collision, intersection, raw_intersection = detector.detect_collision(
        vehicle1_path, vehicle2_path, 
        visualize=True, 
        plot_title="Vehicle Collision Detector"
    )
    
    print(f"Collision detected: {collision}")
    if intersection:
        print(f"Valid intersection point: ({intersection.x:.2f}, {intersection.y:.2f})")
    if raw_intersection and not collision:
        print(f"Lines would intersect at: ({raw_intersection.x:.2f}, {raw_intersection.y:.2f})")

if __name__ == "__main__":
    main()