import cv2
import numpy as np


# define a video capture object
vid = cv2.VideoCapture(0)


def skeleton(img: cv2.Mat):
    done = False
    skel = cv2.Mat(np.zeros((img.shape[0], img.shape[1]), np.uint8))

    while not done:
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)

        temp = cv2.bitwise_not(temp)
        temp = cv2.bitwise_and(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = cv2.erode(img, element)

        _, max, _, _ = cv2.minMaxLoc(img)

        done = max == 0

    return skel


def dist(p1, p2):
    return np.sum((p1 - p2) ** 2) ** 0.5


def is_squarelike(cnt, threshold=0.2, min_area=300):
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False

    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))

    boxarea = cv2.contourArea(box)

    # check boxiness of the derived box
    length = dist(box[0], box[1])
    width = dist(box[1], box[2])

    return abs(boxarea - area) / area < threshold and 0.7 < (length / width) < 1.3


def get_dominant_color(cnt, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    mask = cv2.Mat(mask)

    return cv2.mean(img, mask=mask)


def mangle(a):
    if a > 180:
        a -= 180

    return a


hough_threshold = 100

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    ogframe = cv2.resize(cv2.flip(frame, 1), (740, 360))
    really_ogframe = ogframe

    kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float64)
    # ogframe = cv2.filter2D(ogframe, -1, kernel2)
    ogframe = cv2.bilateralFilter(ogframe, 7, 75, 75)

    # frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2HLS)
    frame = ogframe.astype(np.uint16)
    frame *= 1
    boosted_frame = np.clip(frame, 0, 255).astype(np.uint8)

    kernel = np.zeros((5, 5))

    kernel[2, 2] = 5
    kernel[2, 0] = -1
    kernel[2, 4] = -1
    kernel[0, 2] = -1
    kernel[0, 4] = -1
    # frame = 255 - np.abs(frame) ** 2 // (255)
    # frame = (255 - np.abs(frame)) ** 2 // 5

    # gray_frame = cv2.cvtColor(boosted_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    frame = cv2.Canny(boosted_frame, 70, 120, 5)

    kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(frame, kernel, iterations=10)
    # frame = cv2.dilate(erosion, kernel, iterations=10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame = cv2.dilate(frame, kernel, iterations=1)
    cframe = frame

    # frame = skeleton(frame)

    cnts, hierarchy = cv2.findContours(cframe, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    min_area = 100
    max_area = 1500
    image_number = 0
    for c in cnts:
        continue
        area = cv2.contourArea(c)
        if area > min_area:
            for pt in c:
                pass
    # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, (5, 5))

    points = []
    if hierarchy is not None:

        hierarchy = hierarchy[0]

        for i, c in enumerate(cnts):
            color = (
                np.random.randint(100, 255),
                np.random.randint(100, 255),
                np.random.randint(100, 255),
            )
            if is_squarelike(c):
                color = get_dominant_color(c, ogframe)
                # cv2.drawContours(ogframe, cnts, i, color, thickness=cv2.FILLED)

                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.rectangle(
                    ogframe, (cx - 2, cy - 2), (cx + 2, cy + 2), (0, 0, 0), -1
                )

                points.append((cx, cy))

    valid = True
    # find the largest x and y distance between points
    directions = {}
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                p1, p2 = np.array(points[i]), np.array(points[j])
                diff = p1 - p2

                direction = diff / np.linalg.norm(diff)

                voted = 0

                for other in directions:
                    d1 = direction
                    d2 = np.array(other)

                    dot = np.dot(d1, d2)
                    if dot > 1:
                        dot = 1
                    if dot < -1:
                        dot = -1
                    angle = np.math.acos(dot) * 180 / np.pi

                    if angle > 180:
                        angle -= 180

                    if angle < 10 or angle > 170:
                        # this line matches another direction
                        directions[tuple(other)].append(d1)
                        voted = 1

                    if 80 < angle < 100:
                        directions[tuple(other)].append(d1)
                        voted = 2

                if voted == 0 or voted == 2:
                    directions[tuple(direction)] = []

    # print(directions)

    votes = []
    res = ""
    for d in directions:
        n = len(directions[d])

        votes.append((n, d))

        res += str(d) + " has %s votes, " % n
    # print(res)
    votes = sorted(votes)

    if len(votes) >= 2:
        # x, y = votes[-1][1], votes[-2][1]
        x = votes[-1][1]

        for i in range(len(votes) - 2, -1, -1):
            y = votes[i][1]

            d = np.dot(x, y)

            if -0.05 < d < 0.05:
                # print(d)

                matrix = np.array([x, y]).T
                print(matrix)
                break
        else:
            valid = False

    else:
        valid = False

    if valid:
        best_point = None
        # matrix converts a cube local coordinate to pixel coordinates
        inverse = np.linalg.inv(matrix)

        # find the point that has the most negative coordinates in the cube frame

        for i, candidate in enumerate(points):
            valid = True
            for j, other in enumerate(points):
                if i == j:
                    continue

                vec = (np.array(other) - np.array(candidate)).astype(np.float64)
                vec /= np.linalg.norm(vec)

                dot = np.dot(vec, x / np.linalg.norm(x))
                if dot > 1:
                    dot = 1
                elif dot < -1:
                    dot = -1
                a1 = mangle(np.math.acos(dot) * 180 / np.pi)

                dot = np.dot(vec, y / np.linalg.norm(y))
                if dot > 1:
                    dot = 1
                elif dot < -1:
                    dot = -1
                a2 = mangle(np.math.acos(dot) * 180 / np.pi)

                if a1 > 100 or a2 > 100:
                    valid = False
                # print(cube_coordinate)

            if valid:
                best_point = candidate
                break

        max_x = 0
        max_y = 0
        for p in points:
            vec = np.array(p) - np.array(candidate)
            x_distance = np.dot(vec, x)
            y_distance = np.dot(vec, y)

            if x_distance > max_x:
                max_x = x_distance
            if y_distance > max_y:
                max_y = y_distance

        """
        best_point = None
        for p in points:
            cube_frame = (np.array([p]) @ inverse)[0]

            if best_point is None:
                best_point = cube_frame
                continue

            s = cube_frame[0] + cube_frame[1]

            if s < best_point[0] + best_point[1]:
                best_point = cube_frame
        """

        if best_point is not None:
            # best_point = (best_point @ matrix).astype(int)
            print(best_point)

            corners = np.array(
                [
                    (best_point[0], best_point[1]),
                    (
                        int(best_point[0] + x[0] * max_x),
                        int(best_point[1] + x[1] * max_x),
                    ),
                    (
                        int(best_point[0] + y[0] * max_y + x[0] * max_x),
                        int(best_point[1] + y[1] * max_y + x[1] * max_x),
                    ),
                    (
                        int(best_point[0] + x[0] * max_x),
                        int(best_point[1] + x[1] * max_x),
                    ),
                ]
            )

            cv2.line(
                ogframe,
                (best_point[0], best_point[1]),
                (int(best_point[0] + x[0] * max_x), int(best_point[1] + x[1] * max_x)),
                (0, 255, 0),
                2,
            )

            cv2.line(
                ogframe,
                (best_point[0], best_point[1]),
                (int(best_point[0] + y[0] * max_y), int(best_point[1] + y[1] * max_y)),
                (0, 255, 0),
                2,
            )

            cv2.line(
                ogframe,
                (int(best_point[0] + y[0] * max_y), int(best_point[1] + y[1] * max_y)),
                (
                    int(best_point[0] + y[0] * max_y + x[0] * max_x),
                    int(best_point[1] + y[1] * max_y + x[1] * max_x),
                ),
                (255, 0, 0),
                2,
            )

            cv2.line(
                ogframe,
                (
                    int(best_point[0] + y[0] * max_y + x[0] * max_x),
                    int(best_point[1] + y[1] * max_y + x[1] * max_x),
                ),
                (
                    int(best_point[0] + x[0] * max_x),
                    int(best_point[1] + x[1] * max_x),
                ),
                (255, 0, 0),
                2,
            )

            x = np.array(x)
            y = np.array(y)
            positions = [
                [0, 0],
                x * max_x / 2,
                x * max_x,
                y,
                y * max_y / 2,
                y * max_y / 2,
                x * max_x / 2 + y * max_y / 2,
                x * max_x + y * max_y / 2,
                x * max_x / 2 + y * max_y,
                x * max_x + y * max_y,
            ]

            positions = [list(i) for i in positions]

            positions = np.array(sorted(positions)) + np.array(best_point)

            for p in positions:
                mask = np.zeros(ogframe.shape[:2])
                cv2.circle(mask, p.astype(int), 10, (255, 255, 255), -1)

                mask = cv2.Mat(mask.astype(np.uint8))

                mean_color = cv2.mean(really_ogframe, mask=mask)
                cv2.circle(
                    really_ogframe,
                    p.astype(int) + np.array([200, 0]),
                    10,
                    mean_color,
                    -1,
                )

    """

    new_frame = np.zeros(frame.shape)

    # gframe = cv2.cvtColor(ogframe.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gframe = ogframe

    lines = cv2.HoughLinesP(
        gframe, 5, np.pi / 45, hough_threshold, maxLineGap=20, minLineLength=20
    )

    if lines is not None:
        if len(lines) < 150:
            hough_threshold = max(2, hough_threshold - 1)
        elif len(lines) > 150:
            hough_threshold += 1

        print(len(lines), hough_threshold)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(new_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    """

    # Display the resulting frame
    cv2.imshow("frame", really_ogframe)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
