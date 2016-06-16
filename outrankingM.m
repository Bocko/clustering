function resultat=outrankingM(v1,v2)

[x n]=size(v1);
mat=zeros(n,1);
B=zeros(n,n);
C=zeros(n,n);
r=max(v1);
res=0;
resultat=0;
for i=1:n
    for j=1:n
        if(v1(i)<v1(j) && v2(i)<v2(j) || v1(i)==v1(j) && v2(i)<v2(j) || v1(i)<v1(j) && v2(i)==v2(j) )
            B(i,j)=1;
        end
    end
end
for i=1:n
    for j=1:n
        if(v1(i)<=v1(j) && v2(i)<=v2(j))
            C(i,j)=1;
        end
    end
end


%%On fait une matrice où mat(i,j)=1 lorsque l'élément i appartient à la
%%génération j. On part de la dernière génération (alternatives qui ne
%%dominent aucune autre) puis à l'avant dernière, etc.
%%Cas du dernier cluster
B;
for i=1:n
    if(sum(B(i,:))==0)
      
      mat(i,1)=1; %%(i,1) car c'est le premier groupe pour l'alternative i
    end
end

mat;
m=2;

while(sum(mat(:,m-1)~=0))
    mat(:,m)=zeros(n,1);
    a=find(mat(:,m-1)==1);%a contient l'indice des éléments de la génération précédente
    [num_a lol]=size(a);
    for i=1:n
        for j=1:num_a
            if(sum(i~=a)==num_a && B(i,a(j))==1)
                mat(i,m)=1;
            end
        end
    end
            
    b=find(mat(:,m)==1);
    [num_b lol]=size(b);
    for i=1:num_b
        for j=1:num_b
            if(i~=j && B(b(i),b(j))==1)
                mat(b(i),m)=0;
            end
        end
    end
    m=m+1;
end

nb_groupe=m-2;

g=1;
for i=nb_groupe:-1:1
         a=find(mat(:,i)~=0);
         resultat(a)=nb_groupe-i+1;
end




        
                
    
        
    
   


      
