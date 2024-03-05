import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def image_to_graph(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    G = nx.Graph()
    for contour in contours:
        for point in contour:
            point = tuple(point[0])
            G.add_node(point)
    for contour in contours:
        for i in range(len(contour) - 1):
            point1 = tuple(contour[i][0])
            point2 = tuple(contour[i + 1][0])
            G.add_edge(point1, point2)
    return G, contours

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(frame, (5, 5))
    edges = cv2.Canny(blurred, 50, 150)
    graph, contours = image_to_graph(edges)
    path_frame = frame.copy()

    for contour in contours:
        cv2.drawContours(path_frame, [contour], -1, (255, 0, 0), 2)  # Draw boundaries in blue
        cv2.fillPoly(path_frame, [contour], (0, 255, 0))  # Fill interior with green
    path = list(nx.dfs_preorder_nodes(graph))
    for contour in contours:
        plt.plot(*zip(*contour[:, 0]), 'r')
    plt.plot(*zip(*path), 'bo-')
    plt.gca().invert_yaxis()
    cv2.fillPoly(path_frame, [np.array(path)], (0, 255, 0))
    plt.pause(1)
    plt.clf()

    if len(path) > 0:
        path_img = np.zeros_like(edges)
        path_img = cv2.polylines(path_img, [np.array(path)], isClosed=False, color=(255), thickness=2)
        height, width = path_img.shape[:2]
        segment_height = height // 3
        segment_width = width // 3

        for i in range(3):
            for j in range(3):
                start_row = i * segment_height
                end_row = (i + 1) * segment_height
                start_col = j * segment_width
                end_col = (j + 1) * segment_width
                if i != 1 or j != 1:
                    segment = path_img[start_row:end_row, start_col:end_col]
                    if not np.any(segment == 255):
                        cv2.putText(frame, "path", (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

        cv2.putText(frame, "drone", (width // 2 - 30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        grid_color = (0, 0, 0)
        for i in range(1, 3):
            start_point = (0, i * segment_height)
            end_point = (width, i * segment_height)
            cv2.line(frame, start_point, end_point, grid_color, 1)

        for j in range(1, 3):
            start_point = (j * segment_width, 0)
            end_point = (j * segment_width, height)
            cv2.line(frame, start_point, end_point, grid_color, 1)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
