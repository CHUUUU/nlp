# paper from    
https://arxiv.org/abs/1910.13461   

# data from 
https://github.com/jungyeul/korean-parallel-corpora/tree/master/korean-english-news-v1   

https://github.com/e9t/nsmc   

# Transformer model code from 
https://github.com/jadore801120/attention-is-all-you-need-pytorch

# parameters   
    - epoch 90   
    - batch 32   
    - layer 3   
    - I cut all layer's units in half.
    - GPU Nvidia RTX 2080 (8GB)
 
# pretrain result 
- loss 10 (early step)   
    - inference   
    <bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos><bos>
    
    - label    
    아부 사예프 단체의 고위 지도자라고 말해지는 모빈 압둘라작은 2000년 말레이시아의 한 휴양지에서 21명을 납치하여 수배중이었다   

- loss 9   
    - inference       
    ...........<bos>
    
    - label    
    그는 현재 버지니아 주정부를 상대로 계약 위반으로 소송을 하고 있다.

- loss 8   
    - inference    
    , “ “,,,,,,,,,,,,,,,,,,..,..<bos>
    
    - label    
    이번 경찰의 무장단체 소탕으로 서 수마트라에 위치한 부키팅기의 한 커피전문점을 목표로 한 그들의 테러 음모가 좌절됐다.   
    
- loss 7   
    - inference    
     “,,,,,,,,,에, 위해, 위해 위해 위해 위해 대한 위해 위해 위해 위해 위해 위해,, 대한,, 대한, 위해,,,,에, 수 수 있다.   <bos>   

    - label    
    미국, 러시아, 오스트리아 연구소가 지난해 발굴 된 뼈와 치아 조각에 대한 의학, 법의학, 탄도학 검사를 실시한 결과, 이 유골이 마지막 황제 니
콜라이 2세의 자녀 2명임이 판명됐다.   

- loss 6    
    - inference     
    그는 “이 한 이 더 것을”며 “그러나 이이즈미 것을 있는 이 대한 것을 많은 것을 것이라고고고 말했다.<bos>   
    
    - label     
    그는 “경사스러운 날이었다”며 “모든 가족이 축복하기 위해 모여 주어 큰 힘이 되었었다”고 말했다.    

- loss 5    
    - inference     
    이)에,에에들이 대한할 있는,도 수 있는<eos>,(의,<eos>,’, ‘의,,의,의을 대한,,,,,,, 대한(,,,,,,,,,,,,,, 위해 이,,,,이 것으로 <bos>   

    - label    
    포괄적 규제 대상 품목에는 악기,특대 수화물 가방,여행 가방,핸드백,베이징올림픽 혹은 장애인 올림픽 참가국 및 참가하지 국가들의 국기, 가로 6m 세로 1m 이상의 깃발, 현수막, 전단지, 포스터, 허가 받지 않은 전문가용 비디오 촬영 장비 등이 포함됐다.    

- loss 4    
    - inference     
    코리모의 1(일 비해 이상을 29%     이달 등의 25%됩니다.<bos>   
    
    - label     
    페니매의 주가는 15일 하루에만 27% 하락했고 이달들어 64% 폭락했다.    

- loss 3    
    - inference    
    이나라스스의 애널리스트인 필 플린은 “거래에서부터 “인 “ “ 자한에에 나섰다”며 “이를 전망을 키울이 대해 달러화가 파키    
스탄을 면치 등 나쁜 불가능하다고 계속 나왔다”고 밝혔다. <bos>    
    
    - label     
    알라론트레이딩의 애널리스트인 필 플린은 “거래업자들이 휴일을 앞두고 대대적인 매수청산에 나섰다”며 “취업 전망이 어둡게
나오고 달러화가 약세를 기록하는 등 나쁜 뉴스가 계속 나왔다”고 밝혔다.    

- loss 2
    - inference    
    최근 시위가 과격화 되는 달라고 보여 밝혔으나,,,에서는 이로써 진압 경찰에 밝혀지지 밝혀지지 않았다.<bos>    

    - label      
    최근 시위가 과격화 되는 양상을 보여왔으나 다행히 이날 집회에서는 시위대와 진압 경찰간의 충돌은 발생하지 않았다. 

- loss 1.7   
    - inference   
    영화에서 조리 맥 맥과이어 흉내은 “ “넌 “넌 날 완성시킨다”고 속삭인다.<bos>    
    
    - label   
    영화에서 조커는 제리 맥과이어 흉내를 내며 배트맨에게 “넌 날 완성시킨다”고 속삭인다.   

- loss 1.5
    - inference   
    대만은 7~9월 사이에 태풍의 직간접적인 영향을 받는다.<bos>     
    
    - label   
    대만은 7~9월 사이에 태풍의 직간접적인 영향을 받는다.   

- loss 1.2  (여기서부터 정체시작)
    - inference   
    앙겔라 메르켈 독일 총리 대변인은 불참 입장을 밝혔지만 리처드 슈세이 웨타인마이더 독일 외무장관은 메르켈 총리의 불참이 정치적 항의으로 성격은 아니라고 밝혔다.<bos>      
    
    - label   
    앙겔라 메르켈 독일 총리 또한 불참 입장을 밝혔지만 프랑크 발터 슈타인마이더 독일 외무장관은 메르켈 총리의 불참이 정치적 항의의 성격은 아니라고 밝혔다.   
    
- loss 1.1 
    - inference   
    미국 노스웨스트 에어라인 소속 8 2일 발생한현지시간) 운행 중 문제를 일으켜 승객들이 위험에 노출되는 사고가 발생했다.   
    
    - label   
    미국 노스웨스트 에어라인 소속 항공기가 6일(현지시간) 운행 중 문제를 일으켜 승객들이 위험에 노출되 는 사고가 발생했다.   
    
- loss 1.0    
    - inference    
    터너의 딸 미아 터너는 이번 사건은 대해 받았다고 말했다.   

    - label   
    터너의 딸 미아 터너는 이번 보도에 충격을 받았다고 말했다.   
    
- loss 0.9 (150 epoch)   
    - inference   
    다른 법원들도 변호사들이 변론 취지서를 온라인으로 제출하는 데 허용하고 있다.   

    - label   
    다른 법원들도 변호사들이 변론 취지서를 온라인으로 제출하는 것을 허용하고 있다.   
 
    - inference   
    州州 야키마의 한 법원은 운전자들에게 자신의 그들의하고 그들의 그들의 해명이나 진술을 이메일로 보내는 것을 허용하고 있다.   

    - label   
    워싱턴州 야키마의 한 법원은 운전자들에게 법정에 출두하는 대신 그들의 해명이나 진술을 이메일로 보내는 것을 허용하고 있다.   
    
    
loss 가 0.9 이하로 잘 떨어지지 않는다.   
짧은 길이의 문장은 잘 맞추는 반면, 긴문장은 잘 맞추지 못한다.    
오랜 학습이 필요하거나, 더 많은 데이터, 또는 더 큰 모델이 필요할 것으로 보인다.   


# fine-tuning result (네이버 영화 감정 분류)    
3000 step (32 batch)   
    - acc : 77.775   
6000 step    
    - acc 81.557   
