import numpy as np

def solve_landmarks(clicks_a, clicks_b):
    if len(clicks_a) < 3 or len(clicks_b) < 3:
        print("Need at least 3 landmarks.")
    elif len(clicks_a) != len(clicks_b):
        print("Please select corresponding landmarks alternatively. Ensure equal number of landmarks on both images.")
    elif len(clicks_a) >= 3:
        print('Solving for %d pairs of landmarks'%len(clicks_a))
        x_dash = [] #np.matrix([clicks_b])
        for p in clicks_b:
            x_dash.append(p[0])
            x_dash.append(p[1])
        x_dash = np.matrix(x_dash).reshape(len(x_dash),1)
        X = []
        for p in clicks_a:
            row = [p[0], p[1], 1]
            [row.append(0) for i in range(3)]
            X.append(row)
            row = [0 for i in range(3)]
            row.extend([p[0], p[1], 1])
            X.append(row)
        X = np.matrix(X)
        a = np.linalg.pinv(X) * x_dash
        transformation_matrix = [[a[0,0], a[1,0], a[2,0]],
                                [a[3,0], a[4,0], a[5,0]],
                                [0, 0, 1]]
        return transformation_matrix
