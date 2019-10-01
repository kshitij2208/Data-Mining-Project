data = csvread('CGMDatenumLunchPat1.csv',2,1);
cgm = csvread('CGMSeriesLunchPat1.csv',2,1);
v = NaT(33,31);
w = zeros(33,31);

for i=1:33  
   for j=1:31
       class(data)
       v(i,j) = datetime(data(i,j),'convertFrom','datenum');
       w(i,j) = cgm(i,j);
   end
end
plot(v,w);
 
