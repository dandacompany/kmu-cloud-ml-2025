aws cli 로 awstutor 프로필로 접근해서 워크로드에 배포된 EC2 인스턴스가 있는지 확인해줘.

아주 가벼운 웹서버로 사용할 수 있는 인스턴스 타입을 추천해줘.

> 아래 정책을 삽입하고 진행하세요.

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "VisualEditor0",
			"Effect": "Allow",
			"Action": [
				"iam:CreatePolicy",
                "iam:CreatePolicyVersion",
				"iam:AttachUserPolicy",
				"iam:PutUserPolicy"
			],
			"Resource": "*"
		}
	]
}
```

배포에 필요한 aws cli 용 iam 권한 리스트를 출력해줘.

최우선 추천 인스턴스를 시작하고 https://github.com/dandacompany/kmu_quiz  장고 프로젝트를 배포하고, 
장고 데이터베이스를 postgresql로 교체하기 위해 서버 내부에 설치하고, sqlite 데이터를 postgres로 이관하여 서비스하자.
elasitc ip 를 연결하고 호스팅해줘.

ssh 연결에 문제가 있을때는 라우팅 테이블을 적절히 설정해줘.

collectstatic을 실행하고 정적파일이 잘 호스팅되도록 점검해.



(desec.io로 접속해서 가입후 무료도메인을 발급받으세요)

dante-kmu.dedyn.io 를 a레코드로 고정ip로 매핑했어. caddy 서버를 사용해서 장고서버를 해당 도메인으로 호스팅해줘.
보안그룹 443 포트 확인할것.