function [ firstResult,fineResult,result2 ] = roadDetectNN( inputImage )

    [s1 s2 s3]=size(inputImage);
    s40=300;   %% ilk 300 bant kullanýyorum ben
    imageMean=reshape(inputImage(:,:,1:s40),s1*s2,s40);
    clear inputImage

    for deneyNo=1:5
        deneyNo
        load HyperLibrary.mat   % inside testIndRow and groundTestInd
        [ testIndRow,groundTestInd ] = trainDataReduce(testIndRow, groundTestInd,50);
        testIndRowAll=testIndRow;

        for sizeX=[5 10 20 50]

            [ testIndRow3,groundTestInd ] = trainDataReduce(testIndRow, groundTestInd,sizeX);
            [ testIndRow ] = trainDataReduceOkan( testIndRowAll,sizeX );


            net = patternnet(10);
            [net,tr] = train(net,testIndRow,groundTestInd);
            nntraintool('close')
            sonuc30=net(imageMean');
            ara=vec2ind(sonuc30);
            firstResult{deneyNo}{sizeX}=reshape(ara,s1,s2);

            for x=1:8
                coords=find(groundTestInd(x,:)==1);
                spectra=mean(testIndRow(:,coords)');
                result2(x,:)=hyperSam(imageMean',spectra);
            end

            ourResult=sonuc30./result2;
            ara2=vec2ind(ourResult);
            fineResult{deneyNo}{sizeX}=reshape(ara2,s1,s2);

        end

    end

end
