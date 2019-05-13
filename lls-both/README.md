### 설명
plot.py : history plot용

main.py안의 directory settings.  
```
model_dir : cifar는 ckpt파일 있는 폴더, imagenet은 ckpt파일.  
data_dir : cifar는 데이터 들어있는 폴더, imagenet은 val폴더와 val.txt들어있는 폴더.  
save_dir : history 저장될 폴더 디렉토리 (디렉토리 생성됨).  
np_dir : cifar는 indices numpy file을 외부에 저장해서 따로 지정해줌.  
```

### To do.
1) lls로 풀 batch(or blocks) 고르기. (based on flip된 횟수. 많이 flip된게 high priority).  
2) unary term update. (image pixel마다 most recent marginal gain. u(x=1) = f(x=1|X^(t)) - f(x=-1|X^(t)) & u(x=-1)=0)
3) graph cut 돌려보기. 
