## 이미지 크롭 프로그램

YOLOv3 모델을 사용해 상의/하의/아우터 영역을 크롭한 이미지를 만드는 프로그램입니다.

## 필요 모듈

- tensorflow-gpu
- yacs
- torch

## 사용법

아래 링크에서 yolo.zip 다운로드 후 프로젝트 최상단에 풀어주세요
https://drive.google.com/drive/folders/1rSw1CPdvE8t5SCfjmz11I-TD-igkBQyH

`targets` 폴더에 이미지를 넣고

`python crop_in_dir.py` 를 실행하면

`results` 에 각 이미지를 크롭한 결과가 저장됩니다

결과는 **outer,top,shorts,pants,skirt,dress** 총 6가지가 나올 수 있습니다.

### ex)

![](/images/ex1.PNG)
위 사진처럼 `targets` 폴더에 이미지를 집어넣고 `python crop_in_dir.py`를 실행하면

![](/images/ex2.PNG)
위 사진처럼 `results/[사진 이름]` 폴더에 결과가 저장되고

![](/images/ex3.PNG)
각 폴더에는 크롭된 이미지와 원본 이미지, 크롭한 결과가 `result_txt` 파일에 저장됩니다.

### 참고

참고한 링크: https://www.youtube.com/watch?v=yWwzFnAnrLM
