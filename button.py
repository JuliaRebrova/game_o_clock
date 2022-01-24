import cv2


class Button():
    def __init__(self, pos, text, size, color, start_time=0):
        self.pos = pos
        self.size = size
        self.text = text
        self.color = color
        self.time_step = start_time


    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        colorr = self.color
        cv2.rectangle(img, self.pos, (x + w, y + h), colorr, cv2.FILLED)
        cv2.putText(img, self.text, (x + 10, y + h - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (214, 0, 220), 2)
        return img


    def press(self, palm_r_x, palm_r_y, palm_l_x, palm_l_y):
        x, y = self.pos
        w, h = self.size
        check = ((x <= palm_r_x <= x + w) and (y <= palm_r_y <= y + h)) or ((x <= palm_l_x <= x + w) and (y <= palm_l_y <= y + h))
        return check