**목차**

1. 구현논문
2. 설치
3. 프리트레인모델
4. 본 모델
5. 데이터셋
6. config
7. 논문현황
8. 문제점
9. 개선점
10. 다음 연구원을 위한 조언


# 구현논문
    AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks
    https://arxiv.org/pdf/1711.10485.pdf


# 설치
    git clone https://gitlab.com/SR-Universe/text_to_image_attngan.git
    cd text_to_image_attngan
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install requirement.txt


# 프리트레인모델
    훈련 결과 :  /load_pretrain/output 
    모델 :  load_pretrain/output 에 img_encoder.pth, text_encoder.pth
    재 학습시 : python load_pretain/pretrain.py
    
    간략 설명
    img encoder 와 text encoder 의 유사성을 최대한 올리는 훈련을 함
    훈련평가는 img 와 text 간의 attention 을 시각화 하여 정성적 평가를 함
    모델 용도는 본 모델에 Generator loss 에 포함이 되고, 본 모델 학습시 train=False 임
    논문에서는 The DAMSM loss 로 언급되어 있음
    600 epoch 훈련결과, 정성적 평가로 구현을 성공

    


# 본 모델
    모델 : /output/model/ 에 netD0.pth, netD1.pth, netD2.pth, netG.pth
    재 학습시 python main.py

    간략설명
    Generaotr 가 캐스케이드 멀티스케일 아키텍쳐이고
    Discriminator는 독립적 판별기이다. 
    따라서 Generator 에서 64x64, 128x128, 256x256 이미지를 생성하면,
    Discriminator 는 스케일별, 즉 3개로 독립적으로 있다.
    netG
    netD0 (= 64x64)  
    netD1 (= 128x128)
    netD2 (= 256x256)
    
    
# 데이터셋
    CUB_200_2011 
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    200개의 새의 종의 카테고리가 있고 (하나 카테고리에 하나의 새의 종)
    하나의 카테고리에 한 새의 종에 대한 가변적인 이미지 갯수를 가지고 있고
    1개의 이미지에 10개의 text description 을 가지고 있다.
    즉, text description 으로 데이터 augmentation 되어 있다.
    받기 귀찮다면, 2430 서버에서 data1/Attn_dataset/ 에 birds.tar 를 사용하면 된다.


# config
    안 읽어봐도 됩니다!
    보통 arg + config 를 사용하지만, 한눈에 보는 것을 개인적으로 선호하여 
    하나의 config 에 파일위치, 하이퍼파라미터, 옵션 등등을 전부 정의 함
    config를 변경하여 다시 훈련하고 싶다면 config2.py를 만들기를 추천
    config 에 class_info 를 써 놓았다. (배치 마스킹 용도, 코드 뜯어보면 알수 있다)
    class_info 는 train 과 test 2개로 나뉘어져 있고, 
    숫자는 CUB_200_2011 의 카테고리 번호이고, 
    숫자의 갯수는 해당 카테고리의 가변적 이미지 갯수 이다.
    이것을 코드화 시켰어야 했지만, 버릴수도 있는 부분이라 코드화 하지 않음
    임시로 config에 적어둠



# 논문현황 
    text-to-image
    아키텍쳐별
    
    초기
    (1605) Generative Adversarial Text to Image Synthesis
    (1610) (GAWWN) Learning What and Where to Draw
    
    캐스케이드 멀티스케일 아키텍쳐
    (1611) StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks
    (1710) StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks
    (1711) AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks
    (1903) MirrorGAN: Learning Text-to-image Generation by Redescription
    (1909) Controllable Text-to-Image Generation
    
    하이락키컬 멀티스케일 아키텍쳐
    (1802) (HD-GAN) Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network
    
    듀얼 아키텍쳐
    (1904) (SD-GAN) Semantics Disentangling for Text-to-Image Generation
    (1908) Dual Adversarial Inference for Text-to-Image Synthesis


    작업진행 간략 설명
    메인 아키텍쳐는 캐스케이드 멀티스케일 아키텍쳐 방식이고, 
    17년에 super resolution 또는 deblurring 분야에서 유행하던 하나의 아키텍쳐를 
    현재까지 밀고 있을 정도로 관심도가 굉장히 낮다.
    
    메인 아키텍쳐는 저자가 모두가 중국인으로, 
    기존 연구에 하나, 두개씩 무엇가를 붙여 성능을 향상 시켜왔다
    StackGAN -> StackGAN++ -> AttnGAN  
    AttnGAN -> MirrorGAN 
    AttnGAN -> ControllableGAN
    
    MirrorGAN 과 ControllableGAN 은 모두 AttnGAN을 backborn으로 받아 보강을 했으며,
    ControllableGAN 이 MirrorGAN 을 제외 한건, 
    그 만큼 MirrorGAN 논문이 trash였기 때문이다. (대충 졸업용도 또는 땜빵 느낌의 논문)
    ControllableGAN 이 코드가 공개 되어있지 않아, 구현을 시도하다 접게 된 점을 알려드린다. 
    
    그 외,
    다른 여러 비젼 또는 GAN 논문 히스토리, 즉 년도별 아키텍쳐 변화도를 보면 
    연구원 사이에서 text-to-image 분야의 관심이 거의 없다는 것을 알 수 있다.
    
    
    dialog-to-text
    (1712) CoDraw: Collaborative Drawing as a Testbed for Grounded Goal-driven Communication 
    (1802) ChatPainter: Improving Text to Image Generation using Dialogue
    (1811) Tell, Draw, and Repeat: Generating and Modifying Images
    (1812) StoryGAN: A Sequential Conditional GAN for Story Visualization
    (1812) (SeqAttnGAN) Sequential Attention GAN for Interactive Image Editing via Dialogue
    
    만약 dialog-to-text 를 생각한다면 
    Tell, Draw, and Repeat 을 구현을 올려 놓았으니 참고 바란다.
    
    
# 문제점
    1. Generator가 생성한 이미지가 true space에 들어갈 확률이 적음
        TEXT-TO-IMAGE SYNTHESIS METHOD EVALUATION BASED ON VISUAL PATTERNS
        Firgure 2, 4 참고
        https://arxiv.org/pdf/1911.00077.pdf
    
    2. super resolution 아키텍쳐 문제
        1) AttnGAN에서 사용하는 해상도 올리는 모델은 16년 12월 모델로 오래됨
        deblurring 분야와 super resolution 분야가 비슷한 아키텍쳐를 취하고 있음
        Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring
        https://arxiv.org/pdf/1612.02177.pdf
        
        해상도를 더 좋게 올리는 방법이 현재 많이 제시됨
        
        2) 메모리
        해당 task 가 스케일별 상관성이 있으므로 
        파라미터 sharing 이 가능하다는 점을 고려하면 좋다 
        현재 AttnGan을 16 배치로 24GB 에 다 못올려 CPU 메모리까지 사용 
        1 epoch 당 45분 학습

        3) super sesolution 추론 문제
        Feedback Network for Image Super-Resolution
        Firgure 7 참고
        https://arxiv.org/pdf/1903.09814.pdf
        
    3. Attention 의 문제
        Attention 은 text 로 이미지를 Control 을 가능하게 한다
        Attention 정보로 해상도에 크게 향상 시켜 줌
        
        1. Attention의 역할은 64x64 ⇒ 128x128 로 넘어갈때 일부분이지만 Control 적용을 가능하게 함, (그 일부분엔 우리가 원하는 control 이 포함 안될 수 있다)
        Attention 이 잡아 주는 text 가 명사보다는 거의 잡히지 않고 형용사에 특히 색에 관련되어 잡히는 현상을 포착되었다.
        다만 다행인 것은 형용사가 꾸며주는 명사에 attention 이 잡혀, 형용사에 따라 명사부분이 control 이 된다는 것입니다.   
        그래서 내가 변경하고 싶은 단어에 attention이 잡히지 않으면 control이 되지 않는다.
        왜 attention이 잡히지 않았는가에 대한 논문은 리서치가 많이 필요해 보인다.
        
        2. 가장 이상적인 것은 Attention 이 text가 설명하는 해당 부분에 맞는 부분 이미지만 Attention 이 될 경우 굉장히 이미지가 잘 나옴
        안 좋은 경우는 여러가지가 있지만, 한가지만 예로 들자면, attention이 적용되지 않은 64x64 이미지에는 가지(branch)와 노랑새(yellow bird)가 분리되어 있으나, 
        Attention 이 yellow 라는 단어에서 가지 까지 같이 포함이 되면, 새와 가지와 혼합이 되어, 가지가 새의 일부분으로 표현됨
        즉, Control을 위해 넣은 Attention 의 범위 shape 가 Z의 shape과 상충될때, 망가짐
        반면, Z의 shape 와 attention 범위 shape 가 비슷하면 생성이 잘됨 (그럼에도 이질적인 텍스쳐는 남아 있음)

    
# 개선점
    아키텍쳐 구조 개선에 super resolution 논문들은 해상도 관점을 봄으로 살펴보는 것이 상당히 도움이 된다
    
    CNN 기반 Super Resolution
    (1501) (SRCNN) Image Super-Resolution Using Deep Convolutional Networks
    (1511) (DRCN) Deeply-Recursive Convolutional Network For Image Super-Resolution
    (1511) (VDSR) Accurate Image Super-Resolution Using Very Deep Convolutional Networks
    (1608) (FSRCNN) Accelerating the Super-Resolution Convolutional Neural Network
    (1609) (ESPCN) Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    (17—) (DRRN) Image Super-Resolution via Deep Recursive Residual Network
    (1704) (LapSRN) Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
    (1707) (EDSR, MDSR) Enhanced Deep Residual Networks for Single Image Super-Resolution
    (1712) (ZSSR) "Zero-Shot" Super-Resolution using Deep Internal Learning
    (1712) (SRMD, SRMDNF) Learning a Single Convolutional Super-Resolution Network for Multiple Degradations
    (1802) (DenseSR, RDN) Residual Dense Network for Image Super-Resolution
    (1803) (CARN) Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
    (1803) (DBPN) Deep Back-Projection Networks For Super-Resolution
    (1804) (SRNTT) Reference-Conditioned Super-Resolution by SNeural Texture Transfer
    (1805) (DSRN) Image Super-Resolution via Dual-State Recurrent Networks
    (1806) (NLRN) Non-Local Recurrent Network for Image Restoration
    (1807) (RCAN) Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    (18—) (MSRN) Multi-scale Residual Network for Image
    (1904) (MSRN) Multi-scale deep neural networks for real image super-resolution
    (19—) (SAN) Second-order Attention Network for Single Image Super-Resolution
    (1903) (DPSR) Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels
    (1903) (SRFBN) Feedback Network for Image Super-Resolution
    (1907) (GMFN) Gated Multiple Feedback Network for Image Super-Resolution
    
    
    GAN 기반 Super Resolution
    (1603) (FastNeuralStyle) Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    (1609) (SRResNet, SRGAN) Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    (1612) (EnhanceNet) EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis
    (1702) (EnhanceGAN) Pixel Recursive Super Resolution
    (18—) SRFeat: Single Image Super-Resolution with Feature Discrimination
    (1809) ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    
    
    Attention 개선은 리서치가 필요하다.
    text 와 image 간의 attention 을 많이 보는 분야는 
    text-detection, text-recongnition 이 있다.
    
    GAN 또한 히스토리를 살표보는 것이 좋다.
    
    자연어  
    text 쪽의 전처리를 수정하였다.
    용어가 기억이 안나지만, 
    text 길이가 비슷한것 끼리 배치를 만드는 테크닉을 버리고 
    랜덤 샘플이 가능하게 바꾸었고,
    text 단어 갯수로 통계를 내어 빈도 95% 까지만 word_max_seq 를 잡고 cut을 하였다.
    자연어도 아예 pretrain 하는 것 보다 Elmo 나 Kobert 같은 것을 받아 시도해 보는 것을 추천한다.
    
    
# 다음 연구원을 위한 조언
    text-to-image 가 관심이 없다는 것은 
    적용해 볼 만한것이 넘쳐있다는 뜻이다.
    비록 훈련이 2주단위로 시간이 매우 길겠지만,
    GAN, super resolution, Attention 등 여러 가지를 적용해보고 테스트 해볼 수 있다.
    마지막으로 Controllable Text-to-Image Generation 코드 구현해보길 추천한다
    channel attention을 새로 시도해보고 있고
    특히 (1603) (FastNeuralStyle) Perceptual Losses for Real-Time Style Transfer and Super-Resolution 논문에서 지적하는 
    Z가 사라지는 현상을 해결하기 위한 노력을 한다.
    성능이 올랐는지는 AttnGAN과 비교분석 해보면 좋을 것같다
    Mirror-Gan은 비교용도만 해보고 굳이 코드까지 볼 필요는 없다.
    빠잉!

Based on Continual Linguistic Instruction
    




