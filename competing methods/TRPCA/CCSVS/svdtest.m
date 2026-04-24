addpath(genpath(cd))

clear

pic_name ='.\Test\Pandas.png';
X = double(imread(pic_name));

figure(1)
subplot(1,2,1)
imshow(X/max(X(:)))

X=X/255;

X = fft(X,[],3);
[~,y1,~]= svd(X(:,:,1),'econ');
y1=diag(y1);
[~,y2,~]= svd(X(:,:,2),'econ');
y2=diag(y2);
[~,y3,~]= svd(X(:,:,3),'econ');
y3=diag(y3);
x=1:size(y1,1);
N=100;
x=1:N;

subplot(1,2,2)
imshow(X/max(X(:)))

h=bar(x,[y1(1:N),y2(1:N),y3(1:N)]);
set(h(1),'FaceColor',[1,0,0])
set(h(2),'FaceColor',[0,1,0])
set(h(3),'FaceColor',[0,0,1])
xlim([0,N])
xlabel('The index of sigular values')
ylabel('Magnitde')
legend('Red','Green','Blue')