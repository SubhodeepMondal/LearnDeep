lib_acti= lib/Activations
lib_for_prop= lib/Forward_Propagation
lib_Layers= lib/Layers
lib_math= lib/Math
lib_metrics= lib/Metrics
lib_loss= lib/Losses
lib_models= lib/Models
lib_Opti= lib/Optimizers

inc= include
obj= build/obj
arch= build/archieve
exe= build/exe


compiler= nvcc
archiever = ar

DLearning: src/DLearning.cu $(arch)/libdeeplearning.a 
	$(compiler)  src/DLearning.cu -ldeeplearning -o $(exe)/DLearning.out -I $(inc) -L $(arch) -lcudadevrt -rdc=true
	
$(arch)/libdeeplearning.a: $(obj)/GPULibrary.o $(obj)/MathLibrary.o $(obj)/OptimizationDense.o $(obj)/OptimizationBatchNormalization.o $(obj)/DenseForward.o $(obj)/BatchNormalization.o $(obj)/OptimizationType.o $(obj)/Metrics.o $(obj)/Losses.o $(obj)/Activation.o $(obj)/BatchNormalizationForward.o $(obj)/Layers.o $(obj)/Dense.o $(obj)/BatchNormalization.o $(obj)/Model.o
	$(archiever) cr $(arch)/libdeeplearning.a $(obj)/Activation.o  $(obj)/BatchNormalizationForward.o  $(obj)/BatchNormalization.o  $(obj)/DenseForward.o  $(obj)/Dense.o  $(obj)/GPULibrary.o  $(obj)/Layers.o  $(obj)/Losses.o  $(obj)/MathLibrary.o  $(obj)/Metrics.o  $(obj)/Model.o  $(obj)/OptimizationBatchNormalization.o  $(obj)/OptimizationDense.o  $(obj)/OptimizationType.o

$(obj)/GPULibrary.o: $(lib_math)/GPULibrary.cu $(inc)/GPULibrary.h
	$(compiler) -c $(lib_math)/GPULibrary.cu -o $(obj)/GPULibrary.o  -I include  -lcudadevrt -rdc=true

$(obj)/MathLibrary.o: $(lib_math)/MathLibrary.cu $(inc)/MathLibrary.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_math)/MathLibrary.cu -o $(obj)/MathLibrary.o -I include -lcudadevrt -rdc=true

$(obj)/OptimizationDense.o: $(lib_Opti)/OptimiserDense.cu $(inc)/Optimizers.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_Opti)/OptimiserDense.cu -o $(obj)/OptimizationDense.o -I include -lcudadevrt -rdc=true

$(obj)/OptimizationBatchNormalization.o: $(lib_Opti)/OptimiserBatchNormalization.cu $(inc)/Optimizers.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_Opti)/OptimiserBatchNormalization.cu  -o $(obj)/OptimizationBatchNormalization.o -I include -lcudadevrt -rdc=true

$(obj)/OptimizationType.o: $(lib_Opti)/OptimizationType.cu $(inc)/OptimizationType.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_Opti)/OptimizationType.cu -lcudadevrt -o $(obj)/OptimizationType.o -I include -rdc=true

$(obj)/Metrics.o: $(lib_metrics)/Metrics.cu $(inc)/Metrics.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_metrics)/Metrics.cu -lcudadevrt -o $(obj)/Metrics.o -I include -rdc=true

$(obj)/Losses.o: $(lib_loss)/Losses.cu $(inc)/Losses.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_loss)/Losses.cu -lcudadevrt -o $(obj)/Losses.o -I include -rdc=true

$(obj)/Activation.o: $(lib_acti)/Activations.cu $(inc)/Activations.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_acti)/Activations.cu -lcudadevrt -o $(obj)/Activation.o  -I include -rdc=true

$(obj)/DenseForward.o:$(lib_for_prop)/Dense_Forward.cu $(inc)/Forward_Propagation.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_for_prop)/Dense_Forward.cu -lcudadevrt -o $(obj)/DenseForward.o  -I include -rdc=true

$(obj)/BatchNormalizationForward.o: $(lib_for_prop)/BatchNormalization_Propagation.cu $(inc)/Forward_Propagation.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_for_prop)/BatchNormalization_Propagation.cu -lcudadevrt -o $(obj)/BatchNormalizationForward.o  -I include -rdc=true	

$(obj)/Layers.o: $(lib_Layers)/Layers.cu $(inc)/Layers.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_Layers)/Layers.cu -lcudadevrt -o $(obj)/Layers.o  -I include -rdc=true

$(obj)/Dense.o: $(lib_Layers)/Dense.cu $(inc)/Layers.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_Layers)/Dense.cu -lcudadevrt -o $(obj)/Dense.o  -I include -rdc=true

$(obj)/BatchNormalization.o: $(lib_Layers)/BatchNormalization.cu $(inc)/Layers.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_Layers)/BatchNormalization.cu -lcudadevrt -o $(obj)/BatchNormalization.o  -I include -rdc=true

$(obj)/Model.o: $(lib_models)/Model.cu $(inc)/Model.h $(inc)/NDynamicArray.h
	$(compiler) -c $(lib_models)/Model.cu -lcudadevrt -o $(obj)/Model.o  -I include -rdc=true

clean:
	rm -f $(obj)/*.o $(arch)/*a $(exe)/*.out

clean_obj:
	rm -f $(obj)/*.o

clean_arch:
	rm -f $(arch)/*.a

clean_exe:
	rm -f $(exe)/*.out
