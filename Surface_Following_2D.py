import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def derivative(f, t):
    x, y = f(t)
    dx_dt = (f(t + 1e-9)[0] - x) / 1e-9
    dy_dt = (f(t + 1e-9)[1] - y) / 1e-9
    df_dt = np.array([[dx_dt], [dy_dt]])
    return x, y, df_dt

def rotate_vector(v):
    x, y = v
    v_rotated = np.array([[y], [-x]])  # Rotate 90 degrees counterclockwise
    
    # Check if the rotated vector is pointing downwards (negative y)
    if v_rotated[1, 0] < 0:
        v_rotated = np.array([[-y], [x]])  # Flip the vector to ensure it points upwards
    
    return v_rotated

def angle_between_normal_and_vertical(v_rotated):
    # Assuming normal_vector is a 2D vector (nx, ny)
    nx, ny = v_rotated
    
    # Calculate the angle between the normal vector and the horizontal line
    angle_rad = np.arctan2(ny, nx)
    
    # Adjust the angle to be positive clockwise
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360  # Ensure the angle is positive
    
    return angle_deg

def transform_tcp(angle):
    euler = [0,0,angle] #in degrees
    EAA = [((np.pi*angle)/180),(0,0,1)] #in radiants

    ''' TCP algned with Y axis instead: 
    euler = [0, angle, 0]  # in degrees
    EAA = [0, (0, 1, 0)] 
    '''
    return euler, EAA

def force_at(f,x,y): #force_at(x,y,v_rotated,):
    K = 5 #Stiffness of the surface
    curve_x, curve_y, d = derivative(f, x)

    object_y = y #object pos in Y-axis

    # Calculate the displacement (Δy) of the potential object from the surface
    delta_y = object_y - curve_y
    assert delta_y >= 0, "Object placed lower than surface. Try a higher Y value"
    distance_to_surface = np.abs(delta_y) #distance_to_surface = np.linalg.norm(delta_y)

    # Use the surface normal to determine the direction of the force
    #force_direction = v_rotated / np.linalg.norm(v_rotated)
    #print("Direction of the force made by surface: ", force_direction)

    # Calculate the force using the formula Fy = K * Δy
    Fy = 1 / (K * delta_y)

    return Fy

#CONTROLLERS SECTION:
def follow_surface(x0,y0):
    # Initialize cube state
    position = np.array([x0, y0])
    orientation = 0.5236  # DEFINE HERE INITIAL OBJECT ORIENTATION (radiants)
    Kp = 10

    # Simulation parameters
    dt = 0.1  # Time step
    time_values = []
    position_values = []
    orientation_values = []

    # Simulation loop 
    for i in range(30):

        # Print or use the updated cube state as needed
        print("Time:", i * dt, "Position:", position, "Final Orientation:", np.degrees(orientation))
        time_values.append(i * dt)
        position_values.append(position)
        orientation_values.append(np.degrees(orientation))

        # Get the surface normal at the current position:
        def f(t):
            return np.array([[(t)], [np.sin(t)]]) #DEFINE SURFACE FUNCTION HERE
        x_T, y_T, df_dt = derivative(f, position[0])
        v_rotated = rotate_vector(df_dt)

        # Calculate force exerted by the surface on the cube
        Fy = force_at(f,position[0],position[1])
        
        # Translational control input
        translational_input = Kp * Fy

        # Update cube position
        position += np.squeeze(translational_input * dt * df_dt) #move alongside surface based on tangent vector

        # Rotational control input
        desired_angle = abs((90 - angle_between_normal_and_vertical(v_rotated))) - np.degrees(orientation) #compute correction needed

        orientation += np.radians(desired_angle) #apply correction to re-orient the cube


#-----------------------------------------------------------------------------
follow_surface(0.0,1.0)

'''
How to use the code:

1- Inside follow_surface(x0,y0) function define orientation 
2- Also inside the same function, define the surface function in parametric form. Examples: 
np.array([[(t)], [np.sin(t)]]) curve function, np.array([[(t)], [(1/3*t)]]) straight line
3- Run follow_surface(x,y), giving the starting coordinates of the object. 

The code will print the time, position and orientation for 30 steps (change range of loop to increase data). The object will try to get close to surface 
and balance the force it receives, hence following the surface. 
'''


