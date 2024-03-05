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
    plt.clf()
    nx.draw(graph, with_labels=False, font_weight='bold', node_size=5)
    plt.gca().invert_yaxis()
    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
    path = list(nx.dfs_preorder_nodes(graph))
    for contour in contours:
        plt.plot(*zip(*contour[:, 0]), 'r')
    plt.plot(*zip(*path), 'bo-')
    plt.show(block=False)
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    plt.pause(1)

cap.release()
cv2.destroyAllWindows()
