fileID = fopen('winter_wheat_input_CMC_ifdd_igdd_v20210107_gridUS_4km.csv','r');
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
us_wheat_snow= textscan(fileID, formatSpec, 'Delimiter', ',', 'EmptyValue' ,NaN,'HeaderLines' ,1, 'ReturnOnError', false);
%%us_wheat_snow=csvread('winter_wheat_input_CMC_ifdd_igdd_v20210107_gridUS_4km.csv');

crop_reg=[];
crop_reg(:,1)=log( us_wheat_snow(:,3) ); 
crop_reg(:,2)=  us_wheat_snow(:,3);  
crop_reg(:,3)= us_wheat_snow(:,1) ; %%% county index
crop_reg(:,4)= us_wheat_snow(:,2) ; %%% year number
crop_reg(:,5)=  us_wheat_snow(:, 6);%%%%% fdd1
crop_reg(:,6)=  us_wheat_snow(:,7);%%%%% fdd2
crop_reg(:,7)=  us_wheat_snow(:,8);%%%%% fdd3
crop_reg(:,8)=  us_wheat_snow(:, 17);%%%%% fdd1_sc2_sctf
crop_reg(:,9)= us_wheat_snow(:,32) ;%%%%% gdd1_spring
crop_reg(:,10)=  us_wheat_snow(:,27) ;%%%%% gdd1_winter
crop_reg(:,11)=  us_wheat_snow(:,22) ;%%%%% gdd1_autu
crop_reg(:,12)=  us_wheat_snow(:,33) ;%%%%% gdd2_spring
crop_reg(:,13)=  us_wheat_snow(:,28) ;%%%%% gdd2_winter
crop_reg(:,14)=  us_wheat_snow(:,23) ;%%%%% gdd2_autu
crop_reg(:,15)=  us_wheat_snow(:,34) ;%%%%% gdd3_spring
crop_reg(:,16)=  us_wheat_snow(:,29) ;%%%%% gdd3_winter
crop_reg(:,17)=  us_wheat_snow(:,24);%%%%% gdd3_autu

crop_reg(:,18)=   (us_wheat_snow(:,25)   );%%%%% prcp fall
crop_reg(:,19)=  ( us_wheat_snow(:,30)  );%%%%% prcp winter
crop_reg(:,20)=   ( us_wheat_snow(:,35)  ) ;%%%%% prcp spring

crop_reg(:,21)= (  us_wheat_snow(:,26)) ;%%%%% snow fall
crop_reg(:,22)=  ( us_wheat_snow(:,31) );%%%%% snow winter
crop_reg(:,23)=  ( us_wheat_snow(:,36)  );%%%%% snow spring
crop_reg(:,24)=  us_wheat_snow(:, 5)  ;%%%%% plant area weights

crop_reg(:,25)= irri_sort;%%% irri
crop_reg(:,26)=fert_n_sort; %% fert
crop_reg(:,27)= soil_sand;%%% sand
crop_reg(:,28)=soil_clay; %% clay


data_all=[];
data_all=mat2dataset(crop_reg,'VarNames',{'log_yield','yield','county','year','fdd1','fdd2','fdd3' , 'fdd1_sc2_sctf','gdd1_spring',...
    'gdd1_winter','gdd1_autu', 'gdd2_spring','gdd2_winter','gdd2_autu',   'gdd3_spring','gdd3_winter','gdd3_autu', ...
    'prcp_autu','prcp_winter','prcp_spring','snow_autu','snow_winter','snow_spring' ,'weight','irri_s','fert_s','soil_sand', 'soil_clay'  });

 

%%%%% explicitly consider soil irrigation and fertilizer

lme_y = fitlme(data_all,'log_yield~fdd1+fdd1:fdd1_sc2_sctf+  gdd1_spring+gdd1_winter+gdd1_autu+gdd2_spring+gdd2_winter+gdd2_autu+gdd3_spring+gdd3_winter+gdd3_autu+prcp_autu+prcp_winter+prcp_spring + snow_autu+snow_winter+snow_spring   + fert_s+irri_s +soil_sand+ soil_clay+ (1|county)+(year^2 +year|county)  ' ,'Weights', crop_reg(:,24)  )



 