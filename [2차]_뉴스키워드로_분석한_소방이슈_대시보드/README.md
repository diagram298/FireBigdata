# [2차 : 뉴스키워드로 분석한 소방이슈 대시보드]
![화면 캡처 2023-04-03 155107](https://user-images.githubusercontent.com/93654012/229432813-31157c28-e3aa-4b45-a4c5-bae3b2d4c06d.png)

![1](https://user-images.githubusercontent.com/93654012/229435259-30e3efe5-69de-4111-97a4-807c7eaf0e76.png)
![2](https://user-images.githubusercontent.com/93654012/229435263-0c5988b6-5018-4ec3-a138-4276d6b9b4a7.png)
![3](https://user-images.githubusercontent.com/93654012/229435270-9d7e67a7-4d6d-46d7-9a11-a7593b83855b.png)
![4](https://user-images.githubusercontent.com/93654012/229435278-8270eec1-b011-4668-835c-050daaa7244a.png)
## 1) 기간별 사고 이슈(RED)

1. 소방뉴스 분류 : 
소방뉴스의 기준이란?
    
    1. 연합뉴스 API 데이터 중 “NamedEntity”가 존재 
    2. Class[내용분류]가 `사고` or `자연재해`인 것을 “소방뉴스”로 분류
    
    | Code | 중분류 | Name | Code | 중분류 | Name |
    | --- | --- | --- | --- | --- | --- |
    | 0606001 | 사고 | 사고일반 | 0606008 | 사고 | 안전관리 |
    | 0606002 | 사고 | 육상사고 | 0606009 | 사고 | 자동차사고 |
    | 0606003 | 사고 | 수상사고 | 0606010 | 사고 | 철도사고 |
    | 0606004 | 사고 | 항공사고 | 0606011 | 사고 | 핵사고 |
    | 0606005 | 사고 | 화재방화 | 0606012 | 사고 | 폭발사고 |
    | 0606006 | 사고 | 산업재해 | 0606013 | 사고 | 음주사고 |
    | 0606007 | 사고 | 조난.안전사고 |  |  |  |
    
    | 0607001 | 자연재해 | 자연재해일반 | 0607004 | 자연재해 | 가뭄 |
    | --- | --- | --- | --- | --- | --- |
    | 0607002 | 자연재해 | 풍수해 | 0607005 | 자연재해 | 지진 |
    | 0607003 | 자연재해 | 폭설 | 0607006 | 자연재해 | 태풍 |

1. FROM/TO : “send_timestamp” 컬럼을 “날짜”형태로 태블로 필터에 적용 
    
    
    | News Data - send_timestamp | Tableau - 날짜 |
    | --- | --- |
    | 2022-12-31T23:59:59 | 2022-12-31 |
2. 사고유형 : 해당 기간(FROM/TO) 내 존재하는 사고 유형(ClassCode) 집합을 Dropdown으로 구성

1. 이슈 키워드 :  
    1. NER + 자카드 유사도(Jaccard Similarity) 
    2. 뉴스 제목 클러스터링
        1. 종합 뉴스의 제목에서 키워드들을 추출(Kiwi Analyzer)
            1. 종합 뉴스 : 시간의 흐름에 따라 업데이트 되는 사건 경과에 대한 포괄적인 정보가 포함되어있기 때문에, 한 이슈를 대표할 수 있을만한 뉴스라고 가정 
        2. 추출된 키워드로 본문에 대한 뉴스 클러스터링
        3. 기간 내에 뉴스 빈도수 top5 선정
        
4. 키워드 추이 : 해당 기간 내 일자별로 이슈 키워드 발생 빈도 Line그래프 구성

5. 사건·사고 분류 비율 : 해당 기간 내 존재하는 사고 유형의 비율을 Pie차트로 구성

## 2) 뉴스 분석(YELLOW)

1. 세 줄 요약 : 클러스터링한 결과에서 메인 뉴스를 뽑고, 그 뉴스의 Summarization을 표출
    
    → 메인 뉴스 선정 기준
    
     1. 종합 뉴스가 있는 경우 가장 최근에 나온 종합 뉴스를  메인 뉴스로 선정
    
    ex) 종합 1보, 2보 …. 종합 5보 라면 종합 5보가 메인 뉴스로 선정
    
    1. 종합만 있다면 최초로 나온 종합 기사를 메인 뉴스로 선정
    2. 종합이 없다면 1보, 2보 …. 중에서 가장 최근에 나온 뉴스로 메인 뉴스를 선정
        
        ex) 1보, 2보, 3보 ….라면 3보가 메인 뉴스로 선정
        
    3. 종합이나 1보, 2보가 없다면 최초로 보도된 뉴스로 메인 뉴스를 선정
    
2. 개체명 분석 : WordCloud로 시각화
    1. NAME ENTITY , NER, 워드클라우드
        
        
        | PS | PERSON | AM | ANIMAL |
        | --- | --- | --- | --- |
        | LC | LOCATION | PT | PLANT |
        | OG | ORGANIZATION | QT | QUANTITY |
        | AF | ARTIFACT | FD | STUDY_FIELD |
        | DT | DATE | TR | THEORY |
        | TI | TIME | EV | EVENT |
        | CV | CIVILIZATION | MT | MATERIAL |
        | TM | TERM |  |  |
3. 연관어 분석 : 이슈별 기사들의 ‘Keyword’를 WordCloud 시각화

4. 관련 뉴스
    1. 표시되고 있는 이슈에 해당하는 뉴스 노출
    2. 최신 순으로 정렬



![10](https://user-images.githubusercontent.com/93654012/229435314-f9fb04e1-d25d-45c9-8094-e96a05cc6c4f.png)
![11](https://user-images.githubusercontent.com/93654012/229435319-a58fff0c-4c60-4b1e-b355-6cbd8c3e238e.png)
## 1) 사고 분류별 안전지수지도(RED)

1. 화재안전지수 화재안전지수 산출지표는 2022년 산출식을 기준으로 값을 도출
    
    ![Untitled](https://user-images.githubusercontent.com/93654012/229436364-d4cee158-ab65-47e4-a8bc-b31b87ec3566.png)
    
    1. Choropleth Map을 활용해 화재안전지수 시각화
    
    2. 화재 분야 지역안전지수 산출식 :  
    `100 - (인구만명당 환산 화재사망자수) - (인구만명당 노후건축물수 + 인구만명당 창고 및 운송관련 서비스업 업체수) + (소방정책 예산액 비율) - (인구만명당 화재관련 안전신문고 신고건수 + 기준연도 대비 소소심 교육 인원수)`
        - 위해지표
        인구만명당 환산 화재사망자수 : 0.5 * (0.496 * 1년간 사망자수 + 0.004 * 1년간 발생건수) / 인구수 * 10000
        - 취약지표
        인구만명당 노후건축물수 : 0.0847 * 노후건축물수 / 인구수 * 10000
        인구만명당 창고 및 운송관련 서비스업 업체수 : 0.0153 * 창고 및 운송관련 서비스업 업체수 / 인구-수 * 10000
        - 경감지표
        소방정책 예산액 비율 : 0.2 * (일반회계 + 특별회계 - 특별회계_행정운영경비) / (일반회계 + 특별회계) * 100
        - 의식지표
        인구만명당 화재관련 안전신문고 신고건수 : 0.0346 * 안전신문고 ‘소방안전’ 신고건수 / 인구수 * 10000
        기준연도 대비 소소심 교육 인원수 : 0.1654 * (올해 소소심 교육 인원수-작년 소소심 교육 인원수) / 인구수 * 10000
    3. 데이터 수집처
        1. KOSIS국가통계포털
            1. 시도 별 인구 수
            2. 창고 및 운송 관련 서비스 업체 수
        2. 공공데이터포털
            1. 시도 별 노후 건축물 수 
            2. 소소심 안전 교육 인원 수
            3. 화재 API (화재 발생 건 수 & 사망자 수)
        3. 안전신문고
            1. 시도 별 화재 신고건 수 (일 별 크롤링)
        4. 시도 별 홈페이지
            1. 시도 별 소방 예산
            
2. 화재발생건수 수집 및 해당 기간내 발생건수 시각화

## 2) 연관어 분석 & 사고 원인 분석(YELLOW)

1. 연관어 분석 :  선택된 기간의 소방 화재 뉴스의 ‘NamedEntity’를 WordCloud로 시각화
2. 사고 원인 분석 : MRC 결과에 대해서 라벨링 진행 → 그에 따른 맵핑 결과를 Pie차트로 구성
    
    ![Untitled](https://user-images.githubusercontent.com/93654012/229436827-c0e73a4b-e14c-4851-b6fb-90d229704d81.png)
    

## 3) 관련 뉴스(GREEN)

- [`[1]이슈대시보드`의 관련 뉴스와 동일](https://github.com/diagram298/FireBigdata/tree/main/%5B2%EC%B0%A8%5D_%EB%89%B4%EC%8A%A4%ED%82%A4%EC%9B%8C%EB%93%9C%EB%A1%9C_%EB%B6%84%EC%84%9D%ED%95%9C_%EC%86%8C%EB%B0%A9%EC%9D%B4%EC%8A%88_%EB%8C%80%EC%8B%9C%EB%B3%B4%EB%93%9C#2-%EB%89%B4%EC%8A%A4-%EB%B6%84%EC%84%9Dyellow)
