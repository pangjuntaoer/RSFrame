package prea.main;

import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.DoubleMatrix;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * 信息熵计算
 * 2014年11月5日15:04:13
 * @author Doraemon
 *
 */
public class Comentropy {

	private SparseMatrix ratematrix;
	private int userLength;
	private int itemLength;
	public Comentropy (SparseMatrix ratematrix,int m,int n){
		this.ratematrix = ratematrix;
		this.userLength = m;
		this.itemLength = n;
	}
	
	public void runUserEntropy(){
		long sTime = System.currentTimeMillis();
		for (int i = 0; i < userLength; i++) {
			SparseVector itemVector = ratematrix.getRow(i);
			int itemCount = itemVector.itemCount();
			//DoubleMatrix  matrix = DoubleMatrix.factory.zeros(itemCount,4);//item*4评分
			double data[][]= new double[itemCount][4];
			for (int j = 0; j < itemCount; j++) {
				int valueClass = Double.valueOf(itemVector.getValue(j)).intValue();
				double tast = ratematrix.getMultiValue(i, j, "tast");
				double environment = ratematrix.getMultiValue(i, j, "environment");
				double service = ratematrix.getMultiValue(i, j, "service");
				
				data[j][0]=tast;
				data[j][1]=environment;
				data[j][2]=service;
				data[j][3]=valueClass;
/*				matrix.setAsDouble(tast, j,0);
				matrix.setAsDouble(environment, j,1);
				matrix.setAsDouble(service, j,2);
				matrix.setAsDouble(valueClass, j,3);//类标签,6种 0,1,2,3,4,5
*/			}
			int index = this.getMaxGainIndex(data, itemCount, 6);
			if(index>-1){
				for (int j = 0; j < itemCount; j++) {
					if(index==0){
						double value = ratematrix.getMultiValue(i, j, "tast");
						itemVector.setValue(j, value);
					} else if(index==1){
						double value = ratematrix.getMultiValue(i, j, "environment");
						itemVector.setValue(j, value);
					} else if(index==2){
						double value = ratematrix.getMultiValue(i, j, "service");
						itemVector.setValue(j, value);
					} 
				}
			}
		}
		System.out.println("信息熵计算完成,耗时:"+(System.currentTimeMillis()-sTime)+" ms");
	}
	/**
	 * 计算最大信息增益,取最大增益属性
	 * @return
	 */
	public int getMaxGainIndex(double data[][],int m,int n){
		double total = this.classMeans(data, m, n);
		double gainValue[]=new double[3];
		gainValue[0]=total-this.propertyMeans(0, data, m, n);
		gainValue[1]=total-this.propertyMeans(1, data, m, n);
		gainValue[2]=total-this.propertyMeans(2, data, m, n);
		if(gainValue[0]>gainValue[1]&&gainValue[0]>gainValue[2]){
			return 0;
		}else if(gainValue[1]>gainValue[2]&&gainValue[1]>gainValue[0]){
			return 1;
		}else if(gainValue[2]>gainValue[1]&&gainValue[2]>gainValue[0]){
			return 2;
		}else{
			return -1;
		}
		
	}
	/**
	 * 计算分类期望
	 * @param data
	 * @param m
	 * @param n
	 * @return
	 */
	public double classMeans(double data[][],int m,int n){
		double result = 0d;
		int info[] = new int[6];
		for (int i = 0; i < m; i++) {
			if(data[i][3]==0){
				info[0]++;
			}else if(data[i][3]==1){
				info[1]++;
			}else if(data[i][3]==2){
				info[2]++;
			}else if(data[i][3]==3){
				info[3]++;
			}else if(data[i][3]==4){
				info[4]++;
			}else if(data[i][3]==5){
				info[5]++;
			}
		}
		for (int i = 0; i < info.length; i++) {
			double tmp = 1.0*info[i]/m;
			result+=-(tmp)*Math.log(tmp);
		}
		return result;
	}
	/**
	 * 计算某一个属性的期望信息
	 * @param index
	 * @param data
	 * @param m
	 * @param n
	 * @return
	 */
	public double propertyMeans(int index,double data[][],int m,int n){
		double result = 0d;
		int info[] = new int[6];
		int classValue[][]=new int[6][6];
		for (int i = 0; i < m; i++) {
			int v = Double.valueOf(data[i][3]).intValue();
			if(data[i][index]==0){
				info[0]++;
				classValue[0][v]++;
			}else if(data[i][index]==1){
				info[1]++;
				classValue[1][v]++;
			}else if(data[i][index]==2){
				info[2]++;
				classValue[2][v]++;
			}else if(data[i][index]==3){
				info[3]++;
				classValue[3][v]++;
			}else if(data[i][index]==4){
				info[4]++;
				classValue[4][v]++;
			}else if(data[i][index]==5){
				info[5]++;
				classValue[5][v]++;
			}
		}
		for (int i = 0; i < info.length; i++) {
			double tmp=0;
			for (int j = 0; j < 6; j++) {
				double t =1.0*classValue[i][j]/info[i];
				tmp+=-t*Math.log(t);
			}
			result+=1.0*info[i]/m*tmp;
		}
		return result;
	}
}
