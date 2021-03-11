import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # переменная получающая данные с камеры
imgTarget = cv2.imread('foto10.jpg')  # переменная получающая данные с картинки на которую накладывается видео
myVid = cv2.VideoCapture('video.mp4')  # переменная получающая данные с видео файла

success, imgVideo = myVid.read()  # получаем изображение из файла video.mp4
hT, wT, cT = imgTarget.shape  # создаем переменные будущие координаты для окон
imgVideo = cv2.resize(imgVideo, (wT + 400, hT))  # связываем координаты картинки и видео в один размер

orb = cv2.ORB_create(nfeatures=10)  # заупскаем детектор 1000 раз
kp1, des1 = orb.detectAndCompute(imgTarget, None)  # на изображении imgTarget ищем точки
imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)  # на изображении imgTarget показыввем найденые точки

while True:
    success, imgWebcam = cap.read()  # получаем изображение из файла webcam
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)  # на изображении imgWebcam ищем точки
    imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)  # на изображении imgWebcam показыввем найденые точки
    # imgWebcam = cv2.resize(imgWebcam, (wT, hT))  # связываем координаты картинки и вебки в один размер
        # todo проверить des-ы   купить вебку (на телефоне не работает)
    bf = cv2.BFMatcher()  # созадаем объект для создания совпадений

    # ищем совтпадения  с картинок imgWebcam imgWebcam по параметру des с 2 лучими совпадениями
    macthes = bf.knnMatch(des1, des2, k=2)

    good = []  # пуской список для внесения дистанций М в цикле ниже

    # в этом кикле ищем совпаления в точках М и Н и записываем точки М которые совпали в good на изображениях с
    # imgTarget и imgWebcam
    for m, n in macthes:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))  # видим колличество сопадений
    # ожидаем что списки совпадут совпадения  imgTarget и imgWebcam по kp1  kp2 по совпадениям из списка good
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    # при условии что было найдено более 20 связей (числоможно менять но лучше больше)
    # тогда мы строим рамку вокруг изображения
    if len(good) > 20:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, wT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

    cv2.imshow('img2', img2)  # создаем окно с отображением рамки на imgWebcam
    cv2.imshow('imgFeatures', imgFeatures)  # создаем окно с отображением связей по сопадениям на imgTarget и imgWebcam
    cv2.imshow('imgTarget', imgTarget)  # создаем окно с отображением переменной imgTarget из (foto.jpg)
    cv2.imshow('myVid', imgVideo)  # создаем окно с отображением переменной imgVideo из (video.mp4)
    cv2.imshow('imgWebcam', imgWebcam)  # создаем окно с отображением переменной imgWebcam из (webcam)
    cv2.waitKey(100)
