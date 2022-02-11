rem UPLOAD
kubectl cp tmp/FLEX_RiceGrip imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp



rem Download

rem kubectl cp imori-pod-example:imori-fast-vol/tmp/rollouts/RiceGrip/FLEX/test_2200000 ../Neural_Simulation_PyTorch/tmp/rollouts/RiceGrip/FLEX/test_2200000
rem kubectl cp imori-pod-example:imori-fast-vol/tmp/rollouts/RiceGrip/FLEX2/test_2200000 ../Neural_Simulation_PyTorch/tmp/rollouts/RiceGrip/FLEX2/test_2200000
rem kubectl cp imori-pod-example:imori-fast-vol/tmp/models/RiceGrip/FLEX2/model_600000.pth ../Neural_Simulation_PyTorch/tmp/models/RiceGrip/FLEX2/model_600000.pth


rem kubectl cp imori-pod-example:imori-fast-vol/code/glpointrast/neural_simulation_glpointrast/rollouts ../../glpointrast/glpointrast/neural_simulation_glpointrast/rollouts