# optimizer.zero_grad():
# Pytorch는 gradient를 loss.backward()를 통해 이전 gradient에 누적하여 계산한다.
# 이전 grad 를 누적 시킬 거면 누적 시킬만큼만 사용하고 optimizer.zero_grad()를 사용한다.
# 누적될 필요 없는 모델에서는 model에 input를 통과시키기 전 optimizer.zero_grad()를 한번 호출해 주면 된다.
# 그러면 optimzer 에 준 모든 파라미터에 속성으로 저장되어 있던 grad 가 0이 된다.
