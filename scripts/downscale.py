import os
from PIL import Image
import argparse


# 디렉토리를 입력받아서, 그 디렉토리 안에 x1이라는 폴더 내에 모든 png 이미지를 가로 1/2, 세로 1/2로 줄여서 x2라는 폴더에 저장하는 함수
def downscale(dir):
    
    # x2이라는 폴더를 생성
    os.makedirs(os.path.join(dir, '2x'), exist_ok=True)
    # x1이라는 폴더에 있는 모든 png 파일을 가져옴
    for file in os.listdir(os.path.join(dir, '1x')):
        # 파일의 확장자가 png인 경우
        if file.endswith('.png'):
            # 이미지를 가져옴
            img = Image.open(os.path.join(dir, '1x', file))
            # 이미지의 크기를 1/2로 줄임
            img = img.resize((img.width // 2, img.height // 2))
            # 이미지를 x2 폴더에 저장
            img.save(os.path.join(dir, '2x', file))
            


# main
if __name__ == '__main__':
    
    # argument
    parser = argparse.ArgumentParser()
    # path and static params
    parser.add_argument('--input_dir', type=str, default='./', help='input directory (source views)')
    
    args = parser.parse_args()

    
    # downscale 함수를 실행
    downscale(args.input_dir)
