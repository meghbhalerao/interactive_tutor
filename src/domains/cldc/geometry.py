# geometry.py
# File is originally from the ASTRID paper 
import numpy as np

class Point2D():
    """ Class which represents a point in 2D space."""

    def __init__(self, coordinates=[0, 0]) -> None:
        self.x, self.y = coordinates

    def __eq__(self, other: 'Point2D') -> bool:
        return (self.x==other.x and self.y==other.y)

    def to_vector(self) -> 'Vector2D':
        return Vector2D([self.x, self.y])
    
    def unpack(self) -> tuple:
        return self.x, self.y
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y})'
    
class Point3D():
    """ Class which represents a point in 3D space."""

    def __init__(self, coordinates=[0, 0, 0]) -> None:
        self.x, self.y, self.z = coordinates

    def __eq__(self, other: 'Point3D') -> bool:
        return (self.x==other.x and self.y==other.y and self.z==other.z)

    def to_vector(self) -> 'Vector3D':
        return Vector3D([self.x, self.y, self.z])
    
    def unpack(self) -> tuple:
        return self.x, self.y, self.z
    
    def get_distance_from_plane(self, plane: 'Plane'):
        a, b, c, d = plane.unpack()
        abs_part = np.abs(a*self.x + b*self.y + c*self.z + d)
        sqrt_part = np.sqrt(np.square(a) + np.square(b) + np.square(c))
        return abs_part / sqrt_part
    
    def orthogonal_projection(self, plane: 'Plane') -> 'Point3D':
        a, b, c, d = plane.unpack()
        numerator = a*self.x + b*self.y + c*self.z + d
        denominator = np.square(a) + np.square(b) + np.square(c)
        alpha = - numerator / denominator
        x_prim = self.x + alpha*a
        y_prim = self.y + alpha*b
        z_prim = self.z + alpha*c
        return Point3D([x_prim, y_prim, z_prim])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.z})'

class Vector2D():
    """ Class which represents a vector in 2D space. In addition to Point2D, the
        Vector2D class can execute mathematical operations on vectors: 
            - vector norm,
            - addition and subtraction, 
            - multiplication and division by a scalar, 
            - dot product.
    """

    def __init__(self, components=[0, 0]) -> None:
        self.x, self.y = components

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D([self.x + other.x, self.y + other.y])

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D([self.x - other.x, self.y - other.y])
    
    def __mul__(self, scalar) -> 'Vector2D':
        return Vector2D([self.x*scalar, self.y*scalar])
    
    def __rmul__(self, scalar) -> 'Vector2D':
        return Vector2D([scalar*self.x, scalar*self.y])
    
    def __truediv__(self, scalar) -> 'Vector2D':
        return Vector2D([self.x/scalar, self.y/scalar])
    
    def __floordiv__(self, scalar) -> 'Vector2D':
        return Vector2D([self.x//scalar, self.y//scalar])
    
    def __eq__(self, other: 'Vector2D') -> bool:
        return (self.x==other.x and self.y==other.y)
    
    @staticmethod
    def from_two_points(head: Point2D, tail: Point2D) -> 'Vector2D':
        return head.to_vector() - tail.to_vector()
    
    @staticmethod
    def dot_product(a: 'Vector2D', b: 'Vector2D'):
        return (a.x*b.x + a.y*b.y)
    
    def norm(self):
        return np.sqrt(np.square(self.x) + np.square(self.y))
    
    def to_point(self) -> Point2D:
        return Point2D([self.x, self.y])
    
    def unpack(self) -> tuple:
        return self.x, self.y
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y})'
    
class Vector3D():
    """ Class which represents a vector in 3D space. In addition to Point3D, the
        Vector3D class can execute mathematical operations on vectors: 
            - vector norm,
            - addition and subtraction, 
            - multiplication and division by a scalar, 
            - dot product and cross product.
    """

    def __init__(self, components=[0, 0, 0]) -> None:
        self.x, self.y, self.z = components

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D([self.x + other.x, self.y + other.y, self.z + other.z])

    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D([self.x - other.x, self.y - other.y, self.z - other.z])
    
    def __mul__(self, scalar) -> 'Vector3D':
        return Vector3D([self.x*scalar, self.y*scalar, self.z*scalar])
    
    def __rmul__(self, scalar) -> 'Vector3D':
        return Vector3D([scalar*self.x, scalar*self.y, scalar*self.z])
    
    def __truediv__(self, scalar) -> 'Vector3D':
        return Vector3D([self.x/scalar, self.y/scalar, self.z/scalar])
    
    def __floordiv__(self, scalar) -> 'Vector3D':
        return Vector3D([self.x//scalar, self.y//scalar, self.z//scalar])
    
    def __eq__(self, other: 'Vector3D') -> bool:
        return (self.x==other.x and self.y==other.y and self.z==other.z)
    
    @staticmethod
    def from_two_points(head: Point3D, tail: Point3D) -> 'Vector3D':
        return head.to_vector() - tail.to_vector()

    @staticmethod
    def dot_product(a: 'Vector3D', b: 'Vector3D'):
        return (a.x*b.x + a.y*b.y + a.z*b.z)

    @staticmethod
    def cross_product(a: 'Vector3D', b: 'Vector3D') -> 'Vector3D':
        result = [a.y*b.z - b.y*a.z,
                  -(a.x*b.z - b.x*a.z),
                  a.x*b.y - b.x*a.y]
        return Vector3D(result)
    
    def norm(self):
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))
    
    def to_point(self) -> Point3D:
        return Point3D([self.x, self.y, self.z])
    
    def unpack(self) -> tuple:
        return self.x, self.y, self.z
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.z})'

class Plane():
    """ Class which represents a plane in 3D space.

        __init__ - Class constructor which creates an instance of the 
                                       class Plane from coefficients of the plane equation
                                       a*x + b*y + c*z + d = 0.
        from_normal_vector_and_point - Overload class constructor which creates an instance of the class Plane from a plane normal vector and a point in 3D space given as Poin3D object.
        
        from_three_points - Overload class constructor which creates an instance
                                       of the class Plane from three points in 3D space given
                                       as Point3D objects.
        
        normal_vector - Returns an instance of the class Vector3D which represents
                                       the normal vector of the given Plane object.
                            on_plane - Returns True if the point given as a Point3D object lies
                                       in the plane. Otherwise, it returns False.
                        on_same_side - Returns True if two point given as Point3D objects lie
                                       in the plane. Otherwise, it returns False.
            
    """

    def __init__(self, coefficients=[1, 0, 0, 0]) -> None:
        self.a = coefficients[0]
        self.b = coefficients[1]
        self.c = coefficients[2]
        self.d = coefficients[3]
    
    def __eq__(self, plane: 'Plane') -> bool:
        return (self.a==plane.a and self.b==plane.b and self.c==plane.c and self.d==plane.d)

    @staticmethod
    def from_normal_vector_and_point(normal_vector: Vector3D, point: Point3D) -> 'Plane':
        a, b, c = normal_vector.unpack()
        x, y, z = point.unpack()
        d = -(a*x + b*y + c*z)
        return Plane([a, b, c, d])
    
    @staticmethod
    def from_three_points(A: Point3D, B: Point3D, C: Point3D) -> 'Plane':
        AB = Vector3D.from_two_points(B, A)
        AC = Vector3D.from_two_points(C, A)
        normal_vector = Vector3D.cross_product(AB, AC)
        return Plane.from_normal_vector_and_point(normal_vector, A)

    def normal_vector(self) -> Vector3D:
        return Vector3D([self.a, self.b, self.c])
    
    def on_plane(self, point: Point3D) -> bool:
        x, y, z = point.unpack()
        value = self.a*x + self.b*y + self.c*z + self.d
        return not value
    
    def on_same_side(self, point1: Point3D, point2: Point3D) -> bool:
        x1, y1, z1 = point1.unpack()
        x2, y2, z2 = point2.unpack()
        value1 = self.a*x1 + self.b*y1 + self.c*z1 + self.d
        value2 = self.a*x2 + self.b*y2 + self.c*z2 + self.d
        return (value1 * value2) > 0
    
    def signed_distance(self, point: Point3D) -> float:
        """ Distance (m) from the point to the plane; the sign says which side of the plane the point is on."""
        x, y, z = point.unpack()
        return (self.a*x + self.b*y + self.c*z + self.d) / np.sqrt(self.a**2 + self.b**2 + self.c**2)

    def unpack(self) -> tuple:
        return self.a, self.b, self.c, self.d

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.a}, {self.b}, {self.c}, {self.d})'

def projectPoint(objectPoint: Point3D,
                 rvec: np.ndarray,
                 tvec: np.ndarray,
                 cameraMatrix: np.ndarray,
                 distCoeffs = np.zeros((5,))) -> Point2D:
    """ Project a point in 3D world coordinate space to a pixel on image.

    Args:
         objectPoint: Point3D point object that shold be projected to the image.
                rvec: The rotation matrix that, together with tvec, performs a change 
                      of basis from world to camera coordinate system.
                tvec: The translation vector.
        cameraMatrix: Intrinsic camera matrix.
          distCoeffs: Camera distortion coefficients [k1, k2, p1, p2, k3].

    Returns:
        A list of touples representing the projected points into pixels.
    """

    # Convert to Camera coordinate system
    point_word_x, point_word_y, point_word_z = objectPoint.unpack() 
    point_word_homogeneous = np.array([point_word_x, point_word_y, point_word_z, 1])
    point_camera = np.concatenate((rvec, np.reshape(tvec, (3,1))), axis=1) @ point_word_homogeneous

    # Normalize
    point_camera_x, point_camera_y, point_camera_z = point_camera
    x_prime = point_camera_x / point_camera_z
    y_prime = point_camera_y / point_camera_z

    # Distortion
    k1, k2, p1, p2, k3 = distCoeffs
    r_square = np.square(x_prime) + np.square(y_prime)
    x_secundum = x_prime*(1 + k1*r_square + k2*r_square**2 + k3*r_square**3) + 2*p1*x_prime*y_prime + p2*(r_square + 2*x_prime**2)
    y_secundum = y_prime*(1 + k1*r_square + k2*r_square**2 + k3*r_square**3) + p1*(r_square + 2*y_prime**2) + 2*p2*x_prime*y_prime

    # Pixel
    u, v, _ = cameraMatrix @ np.array([x_secundum, y_secundum, 1])

    return Point2D(np.array([u, v], dtype=np.int16))

def unprojectPoint(imagePoint: Point2D, 
                   pointDepth: np.int32,
                   cameraMatrix: np.ndarray) -> Point3D:
    """ Projects a pixel into 3D Camera coordinate space.

    Args:
          imagePoint: Pixel that shold be projected into 3D camera coordinate system.
               depth: Depth of the pixel, z coordinate in 3D camera coordinate system.
        cameraMatrix: Intrinsic camera matrix.
          distCoeffs: Camera distortion coefficients [k1, k2, p1, p2, k3].

    Returns:
        Point3D object representing the projected pixel.
    """
    
    # Convert from pixel to 3D point in Camera coordinate system
    u, v = imagePoint.unpack()
    fx, fy = cameraMatrix[0][0], cameraMatrix[1][1]
    cx, cy = cameraMatrix[0][2], cameraMatrix[1][2]
    point_camera_z = pointDepth
    point_camera_x = (u - cx) * point_camera_z / fx
    point_camera_y = (v - cy) * point_camera_z / fy

    return Point3D([point_camera_x, point_camera_y, point_camera_z])


