# MILESTONE
This is Aiffel MILESTONE Team project
## Project : AI 기반 리튬이온배터리용 불연성 첨가제 설계 및 역합성을 통한 반응물 도출
## 소개
- 	전해질 첨가제는 음양극 표면에 보호막 형성, 과충전 방지, 전도성 향상 등 리튬이온 배터리의 수명과 안정성을 확보하기 위한 전해질 제조 과정의 필수적인 요소임.
-	기존의 trial-error 실험방식은 인적물적 자원 소모적이며 신물질 개발에 오랜 기간이 소요됨. 양자계산을 이용한 방식은 물적 자원을 소모하지 않고 기간을 단축시킬 수는 있으나, 대용량 물질 스크리닝에는 적합하지 않음. 
-	이에 AI 기법을 활용하여 자원 소모를 최소화함과 동시에 효율적으로 불연 특성을 갖는 유기물 구조체 (첨가제) 생성 모델을 설계하고자 함. 이후 역합성을 진행하여 반응물(reactant)을 도출함으로써 사람의 노력이 들어가는 부분을 최소화하고자 함.


## 전체 워크플로우 구성 및 설명
-	전체적인 워크플로우 구성은 그림1을 통해 확인 가능함.
-	SMILES code로 분자를 구현한 데이터셋을 그래프 표현으로 변환하여 생성모델에서 학습함.
-	Novelity, Validity가 1.0이 되는 지점에서 분자를 2만개 생성함. 이때 SAS(합성가능성지표)를 3.5 이상을 충족하도록 컨디션을 걸어줌.
-	Uniqueness 지표는 0.7을 기준으로 잡음
-	생성 모델(Generation Model)을 거쳐 새로운 유기분자 DB 를 생성함.

-	전처리된 DB는 스크리닝 모델(Screening Model)을 거쳐 우수 후보군(New Additives)을 선정 함.
-	스크리닝 모델은 3개의 선별자(Descriptor)로 이루어져 있음; (1) HOMO 에너지 (2) LUMO 에너지 (3) Dipole moment(쌍극자 모멘트)
-	각 선별자는 그래프 기반 GCN 모델을 이용해 미리 분자의 특성을 사전학습하고, 순차적으로 쌓아 올림으로써 스크리닝 모델을 구성함.
-	각 선별자로 모델 initialization 시에 100, 200, 300 seed에서 각각 학습하여 pretrained model 총 9개를 생성함.
-	저장된 pretrained model을 모두 불러와 앙상블 기법으로 각 선별자를 예측함
-	스크리닝 모델을 통해 최종 선별된 우수 후보군은 역합성 모델로 전달됨.
  
-	역합성 모델은 생성된 유기분자를 분해하여 반응물(Reactants)을 도출함.
-	역합성 후 도출된 반응물들은 InchI Key로 변환되어 e-molecule dataset에 동일한 분자가 있는지 체크함.
-	있다면 역합성 완료. 없다면 Recursive iteration을 돌며 추가 역합성을 진행함.
-	최대 iteration depth는 5로 지정함. 

<div align='center'>
<image src='[https://github.com/LubyJ/MILESTONE/blob/main/images/Workflow.png' width='80%' height='80%' alt='workflow'><br>
전체 워크플로우 구성
</div>
