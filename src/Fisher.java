import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

import java.util.*;


public class Fisher {
	public static void main(String[] args){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		double[] w1 = new double[]{-0.4,0.58,0.089,
								   -0.31,0.27,-0.04,
								   0.38,0.055,-0.035,
								   -0.15,0.53,0.011,
								   -0.35,0.47,0.034,
								   0.17,0.69,0.1,
								   -0.011,0.55,-0.18,
								   -0.27,0.61,0.12,
								   -0.065,0.49,0.0012,
								   -0.12,0.054,-0.063};
		double[] w2 = new double[]{0.83,1.6,-0.014,
								   1.1,1.6,0.48,
								   -0.44,-0.41,0.32,
								   0.047,-0.45,1.4,
								   0.28,0.35,3.1,
								   -0.39,-0.48,0.11,
								   0.34,-0.079,0.14,
								   -0.3,-0.22,2.2,
								   1.1,1.2,-0.46,
								   0.18,-0.11,-0.49};
		
		MatOfDouble m1= new MatOfDouble();
		MatOfDouble m2 = new MatOfDouble();
		
		m1.fromArray(w1);
		m2.fromArray(w2);
		
		Mat mw1 = new Mat();
		Mat mw2 = new Mat();
		
		mw1 = m1.reshape(1, 10);
		mw2 = m2.reshape(1, 10);
		
		double sum1 = 0, sum2 =0, sum3 = 0;
		
		for(int i=0; i<mw1.rows(); i++){
			for(int j=0; j<mw1.cols(); j++){
				double[] temp = mw1.get(i, j);
				switch(j){
					case 0: sum1 = sum1 +temp[0];break;
					case 1: sum2 = sum2 +temp[0];break;
					case 2: sum3 = sum3 +temp[0];break;
				}
			}
		}
		int n = mw1.rows();
		MatOfDouble sum_w1 = new MatOfDouble(sum1/n, sum2/n, sum3/n);
		
		sum1 = 0;
		sum2 =0;
		sum3 = 0;
		
		for(int i=0; i<mw2.rows(); i++){
			for(int j=0; j<mw2.cols(); j++){
				double[] temp = mw2.get(i, j);
				switch(j){
					case 0: sum1 = sum1 +temp[0];break;
					case 1: sum2 = sum2 +temp[0];break;
					case 2: sum3 = sum3 +temp[0];break;
				}
			}
		}
		int n2 = mw2.rows();
		MatOfDouble sum_w2 = new MatOfDouble(sum1/n2, sum2/n2, sum3/n2);
		
		Mat ave_w1 = new Mat(), ave_w2 = new Mat();
		ave_w1 = sum_w1.reshape(1,1);
		ave_w2 = sum_w2.reshape(1,1);

		double x1=0, x2=0, x3=0;
		Mat sw1 = new Mat(3,3,CvType.CV_64F);
		for(int i=0; i<mw1.rows(); i++){
			for(int j=0; j<mw1.cols(); j++){
				double[] temp = mw1.get(i, j);
				switch(j){
				case 0: x1 = temp[0];break;
				case 1: x2 = temp[0];break;
				case 2: x3 = temp[0];break;
				}
			}
			MatOfDouble xm = new MatOfDouble(x1, x2, x3);
			Mat nxm = new Mat();
			nxm = xm.reshape(1,1);
			Mat result = new Mat(), tresult = new Mat();
			Core.subtract(nxm, ave_w1, result);
			Core.transpose(result, tresult);
			matMul(tresult, result, result);
			Core.add(result, sw1, sw1);
		}

		Mat sw2 = new Mat(3,3,CvType.CV_64F);
		for(int i=0; i<mw2.rows(); i++){
			for(int j=0; j<mw2.cols(); j++){
				double[] temp = mw2.get(i, j);
				switch(j){
				case 0: x1 = temp[0];break;
				case 1: x2 = temp[0];break;
				case 2: x3 = temp[0];break;
				}
			}
			MatOfDouble xm2 = new MatOfDouble(x1, x2, x3);
			Mat nxm2 = new Mat();
			nxm2 = xm2.reshape(1,1);
			Mat result2 = new Mat(), tresult2 = new Mat();
			Core.subtract(nxm2, ave_w2, result2);
			Core.transpose(result2, tresult2);
			matMul(tresult2, result2, result2);
			Core.add(result2, sw2, sw2);
		}
		
		Mat sw = new Mat(3, 3, CvType.CV_64F);
		Core.add(sw1, sw2, sw);
		
		Mat tsw = new Mat(), sub_w1w2 = new Mat(), w = new Mat();
		Core.transpose(sw, tsw);
		Core.subtract(ave_w1, ave_w2, sub_w1w2);
		Core.transpose(sub_w1w2, sub_w1w2);
		matMul(tsw, sub_w1w2, w);
		System.out.println(w.dump());
		
	}
	public static Mat matMul(Mat A, Mat B, Mat C){
		Core.gemm(A, B, 1.0, Mat.zeros(A.size(), A.type()), 0.0, C);
		return C;
	}
	
}
