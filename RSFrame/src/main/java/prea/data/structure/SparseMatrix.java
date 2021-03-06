package prea.data.structure;

import java.io.Serializable;
/**
 * This class implements sparse matrix, containing empty values for most space.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class SparseMatrix implements Serializable{
	private static final long serialVersionUID = 8003;
	
	/** The number of rows. */
	private int M;
	/** The number of columns. */
	private int N;
	/** The array of row references. */
	private SparseVector[] rows;
	/** The array of column references. */
	private SparseVector[] cols;

	
	/**
	 * 新增字段，存储其他评分
	 */
	private double tast[][];
	private double environment[][];
	private double service[][];
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 *针对多评分的构造函数
	 * @param m
	 * @param n
	 * @param multiRate
	 */
	public SparseMatrix(int m, int n,boolean multiRate) {
		this.M = m;
		this.N = n;
		rows = new SparseVector[M];
		cols = new SparseVector[N];
		tast = new double[m][n];
		environment = new double[m][n];
		service = new double[m][n];
		for (int i = 0; i < M; i++) {
			rows[i] = new SparseVector(N);
		}
		for (int j = 0; j < N; j++) {
			cols[j] = new SparseVector(M);
		}
	}
	/**
	 * 设置值
	 * @param i
	 * @param j
	 * @param value
	 * @param tastRate
	 * @param environmentRate
	 * @param serviceRate
	 * @throws Exception 
	 */
	public void setMultiValue(int i, int j, double value,double tastRate,double environmentRate,double serviceRate) throws Exception {
		if(environment==null){
			throw new Exception("该方法必须是用SparseMatrix(int m, int n,boolean multiRate) 构造方法的对象");
		}
		if (value == 0.0) {
			rows[i].remove(j);
			cols[j].remove(i);
		}
		else {
			rows[i].setValue(j, value);
			cols[j].setValue(i, value);
			tast[i][j] = tastRate;
			environment[i][j] = environmentRate;
			service [i][j] = serviceRate;
		}
	}
	/**
	 * 获取多评分中指定的评分
	 * @param i
	 * @param j
	 * @param type
	 * @return
	 */
	public double getMultiValue(int i,int j,String type){
		if("tast".equals(type)){
			return tast[i][j];
		}else if("environment[i][j]".equals(type)){
			return environment[i][j];
		}else if("service".equals(type)){
			return service[i][j];
		}
		return 0d;
	}
	/**
	 * 清空多余评分内存
	 */
	public void clearMultiRate(){
		tast = null;
		environment = null;
		service = null;
	}
	
	/**
	 * Construct an empty sparse matrix, with a given size.
	 * 
	 * @param m The number of rows.
	 * @param n The number of columns.
	 */
	public SparseMatrix(int m, int n) {
		this.M = m;
		this.N = n;
		rows = new SparseVector[M];
		cols = new SparseVector[N];
		
		for (int i = 0; i < M; i++) {
			rows[i] = new SparseVector(N);
		}
		for (int j = 0; j < N; j++) {
			cols[j] = new SparseVector(M);
		}
	}
	
	/**
	 * Construct an empty sparse matrix, with data copied from another sparse matrix.
	 * 
	 * @param sm The matrix having data being copied.
	 */
	public SparseMatrix(SparseMatrix sm) {
		this.M = sm.M;
		this.N = sm.N;
		rows = new SparseVector[M];
		cols = new SparseVector[N];
		
		for (int i = 0; i < M; i++) {
			rows[i] = sm.getRow(i);
		}
		for (int j = 0; j < N; j++) {
			cols[j] = sm.getCol(j);
		}
	}

	/*========================================
	 * Getter/Setter
	 *========================================*/
	/**
	 * Retrieve a stored value from the given index.
	 * 
	 * @param i The row index to retrieve.
	 * @param j The column index to retrieve.
	 * @return The value stored at the given index.
	 */
	public double getValue(int i, int j) {
		return rows[i].getValue(j);
	}
	
	/**
	 * Set a new value at the given index.
	 * 
	 * @param i The row index to store new value.
	 * @param j The column index to store new value.
	 * @param value The value to store.
	 */
	public void setValue(int i, int j, double value) {
		if (value == 0.0) {
			rows[i].remove(j);
			cols[j].remove(i);
		}
		else {
			rows[i].setValue(j, value);
			cols[j].setValue(i, value);
		}
	}
	
	/**
	 * Return a reference of a given row.
	 * Make sure to use this method only for read-only purpose.
	 * </br>��õ�index������
	 * @param index The row index to retrieve.
	 * @return A reference to the designated row.
	 */
	public SparseVector getRowRef(int index) {
		return rows[index];
	}
	
	/**
	 * Return a copy of a given row.
	 * Use this if you do not want to affect to original data.
	 * 
	 * @param index The row index to retrieve.
	 * @return A reference to the designated row.
	 */
	public SparseVector getRow(int index) {
		SparseVector newVector = this.rows[index].copy();
		
		return newVector;
	}
	
	/**
	 * Return a reference of a given column.
	 * Make sure to use this method only for read-only purpose.
	 * </br>��õ�index������
	 * @param index The column index to retrieve.
	 * @return A reference to the designated column.
	 */
	public SparseVector getColRef(int index) {
		return cols[index];
	}
	
	/**
	 * Return a copy of a given column.
	 * Use this if you do not want to affect to original data.
	 * 
	 * @param index The column index to retrieve.
	 * @return A reference to the designated column.
	 */
	public SparseVector getCol(int index) {
		SparseVector newVector = this.cols[index].copy();
		
		return newVector;
	}
	
	/**
	 * Convert the matrix into the array-based dense representation.
	 * 
	 * @return The dense version of same matrix.
	 */
	public DenseMatrix toDenseMatrix() {
		DenseMatrix m = new DenseMatrix(this.M, this.N);
		
		for (int i = 0; i < this.M; i++) {
			int[] indexList = this.getRowRef(i).indexList();
			if (indexList != null) {
				for (int j : indexList) {
					m.setValue(i, j, this.getValue(i, j));
				}
			}
		}
		
		return m;
	}
	
	/**
	 * Convert the matrix into the array-based dense representation,
	 * but only with the selected indices.
	 * 
	 * @param indexList The list of indices converting to dense matrix.
	 * @return The dense representation of same matrix, with given indices. 
	 */
	public DenseMatrix toDenseSubset(int[] indexList) {
		if (indexList == null || indexList.length == 0)
			return null;
			
		DenseMatrix m = new DenseMatrix(indexList.length, indexList.length);
		
		int x = 0;
		for (int i : indexList) {
			int y = 0;
			for (int j : indexList) {
				m.setValue(x, y, this.getValue(i, j));
				y++;
			}
			x++;
		}
		
		return m;
	}
	
	/**
	 * Convert the matrix into the array-based dense representation,
	 * but only with the selected indices, both rows and columns separately.
	 * 
	 * @param rowList The list of row indices converting to dense matrix.
	 * @param colList The list of column indices converting to dense matrix.
	 * @return The dense representation of same matrix, with given indices. 
	 */
	public DenseMatrix toDenseSubset(int[] rowList, int[] colList) {
		if (rowList == null || colList == null || rowList.length == 0 || colList.length == 0)
			return null;
			
		DenseMatrix m = new DenseMatrix(rowList.length, colList.length);
		
		int x = 0;
		for (int i : rowList) {
			int y = 0;
			for (int j : colList) {
				m.setValue(x, y, this.getValue(i, j));
				y++;
			}
			x++;
		}
		
		return m;
	}

	/*========================================
	 * Properties
	 *========================================*/
	/**
	 * Capacity of this matrix.
	 * 
	 * @return An array containing the length of this matrix.
	 * Index 0 contains row count, while index 1 column count.
	 */
	public int[] length() {
		int[] lengthArray = new int[2];
		
		lengthArray[0] = this.M;
		lengthArray[1] = this.N;
		
		return lengthArray;
	}

	/**
	 * Actual number of items in the matrix.
	 * 
	 * @return The number of items in the matrix.
	 */
	public int itemCount() { 
		int sum = 0;
		
		if (M > N) {
			for (int i = 0; i < M; i++) {
				sum += rows[i].itemCount();
			}
		}
		else {
			for (int j = 0; j < N; j++) {
				sum += cols[j].itemCount();
			}
		}
		
		return sum;
	}
	
	/**
	 * Set a new size of the matrix.
	 * 
	 * @param m The new row count.
	 * @param n The new column count.
	 */
	public void setSize(int m, int n) {
		this.M = m;
		this.N = n;
	}
	
	/**
	 * Return items in the diagonal in vector form.
	 * 
	 * @return Diagonal vector from the matrix.
	 */
	public SparseVector diagonal() {
		SparseVector v = new SparseVector(Math.min(this.M, this.N));
		
		for (int i = 0; i < Math.min(this.M, this.N); i++) {
			double value = this.getValue(i, i);
			if (value > 0.0) {
				v.setValue(i, value);
			}
		}
		
		return v;
	}
	
	/**
	 * The value of maximum element in the matrix.
	 * 
	 * @return The maximum value.
	 */
	public double max() {
		double curr = Double.MIN_VALUE;
		
		for (int i = 0; i < this.M; i++) {
			SparseVector v = this.getRowRef(i);
			if (v.itemCount() > 0) {
				double rowMax = v.max();
				if (v.max() > curr) {
					curr = rowMax;
				}
			}
		}
		
		return curr;
	}
	
	/**
	 * The value of minimum element in the matrix.
	 * 
	 * @return The minimum value.
	 */
	public double min() {
		double curr = Double.MAX_VALUE;
		
		for (int i = 0; i < this.M; i++) {
			SparseVector v = this.getRowRef(i);
			if (v.itemCount() > 0) {
				double rowMin = v.min();
				if (v.min() < curr) {
					curr = rowMin;
				}
			}
		}
		
		return curr;
	}
	
	/**
	 * Sum of every element. It ignores non-existing values.
	 * 
	 * @return The sum of all elements.
	 */
	public double sum() {
		double sum = 0.0;
		
		for (int i = 0; i < this.M; i++) {
			SparseVector v = this.getRowRef(i);
			sum += v.sum();
		}
		
		return sum;
	}
	
	/**
	 * Average of every element. It ignores non-existing values.
	 * 
	 * @return The average value.
	 */
	public double average() {
		return this.sum() / this.itemCount();
	}
	
	/**
	 * Variance of every element. It ignores non-existing values.
	 * 
	 * @return The variance value.
	 */
	public double variance() {
		double avg = this.average();
		double sum = 0.0;
		
		for (int i = 0; i < this.M; i++) {
			int[] itemList = this.getRowRef(i).indexList();
			if (itemList != null) {
				for (int j : itemList) {
					sum += Math.pow(this.getValue(i, j) - avg, 2);
				}
			}
		}
		
		return sum / this.itemCount();
	}
	
	/**
	 * Standard Deviation of every element. It ignores non-existing values.
	 * 
	 * @return The standard deviation value.
	 */
	public double stdev() {
		return Math.sqrt(this.variance());
	}
	
	/*========================================
	 * Matrix operations
	 *========================================*/
	/**
	 * Scalar subtraction (aX).
	 * </br>һ���������*һ������
	 * @param alpha The scalar value to be multiplied to this matrix.
	 * @return The resulting matrix after scaling.
	 */
	public SparseMatrix scale(double alpha) {
		SparseMatrix A = new SparseMatrix(this.M, this.N);
		
		for (int i = 0; i < A.M; i++) {
			A.rows[i] = this.getRowRef(i).scale(alpha);
		}
		for (int j = 0; j < A.N; j++) {
			A.cols[j] = this.getColRef(j).scale(alpha);
		}
		
		return A;
	}
	
	/**
	 * Scalar subtraction (aX) on the matrix itself.
	 * This is used for minimizing memory usage.
	 * 
	 * @param alpha The scalar value to be multiplied to this matrix.
	 */
	public void selfScale(double alpha) {
		for (int i = 0; i < this.M; i++) {
			int[] itemList = this.getRowRef(i).indexList();
			if (itemList != null) {
				for (int j : itemList) {
					this.setValue(i, j, this.getValue(i, j) * alpha);
				}
			}
		}
	}
	
	/**
	 * Scalar addition.
	 * @param alpha The scalar value to be added to this matrix.
	 * @return The resulting matrix after addition.
	 */
	public SparseMatrix add(double alpha) {
		SparseMatrix A = new SparseMatrix(this.M, this.N);
		
		for (int i = 0; i < A.M; i++) {
			A.rows[i] = this.getRowRef(i).add(alpha);
		}
		for (int j = 0; j < A.N; j++) {
			A.cols[j] = this.getColRef(j).add(alpha);
		}
		
		return A;
	}
	
	/**
	 * Scalar addition on the matrix itself.
	 * @param alpha The scalar value to be added to this matrix.
	 */
	public void selfAdd(double alpha) {
		for (int i = 0; i < this.M; i++) {
			int[] itemList = this.getRowRef(i).indexList();
			if (itemList != null) {
				for (int j : itemList) {
					this.setValue(i, j, this.getValue(i, j) + alpha);
				}
			}
		}
	}
	
	/**
	 * Exponential of a given constant.
	 * 
	 * @param alpha The exponent.
	 * @return The resulting exponential matrix.
	 */
	public SparseMatrix exp(double alpha) {
		for (int i = 0; i < this.M; i++) {
			SparseVector b = this.getRowRef(i);
			int[] indexList = b.indexList();
			
			if (indexList != null) {
				for (int j : indexList) {
					this.setValue(i, j, Math.pow(alpha, this.getValue(i, j)));
				}
			}
		}
		
		return this;
	}
	
	/**
	 * The transpose of the matrix.
	 * This is simply implemented by interchanging row and column each other. 
	 * <br> �����������ת�þ���
	 * @return The transpose of the matrix.
	 */
	public SparseMatrix transpose() {
		SparseMatrix A = new SparseMatrix(this.N, this.M);
		
		A.cols = this.rows;
		A.rows = this.cols;
		
		return A;
	}
	
	/**
	 * Matrix-vector product (b = Ax)
	 * </br>��������������ĳ˻�=����
	 * @param x The vector to be multiplied to this matrix.
	 * @throws RuntimeException when dimensions disagree
	 * @return The resulting vector after multiplication.
	 */
	public SparseVector times(SparseVector x) {
		if (N != x.length())
			throw new RuntimeException("Dimensions disagree");
		
		SparseMatrix A = this;
		SparseVector b = new SparseVector(M);
		
		for (int i = 0; i < M; i++) {
			b.setValue(i, A.rows[i].innerProduct(x));
		}
		
		return b;
	}
	
	/**
	 * Matrix-matrix product (C = AB)
	 * 
	 * @param B The matrix to be multiplied to this matrix.
	 * @throws RuntimeException when dimensions disagree
	 * @return The resulting matrix after multiplication.
	 */
	public SparseMatrix times(SparseMatrix B) {
		// original implementation
		if (N != (B.length())[0])
			throw new RuntimeException("Dimensions disagree");
		
		SparseMatrix A = this;
		SparseMatrix C = new SparseMatrix(M, (B.length())[1]);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < (B.length())[1]; j++) {
				SparseVector x = A.getRowRef(i);
				SparseVector y = B.getColRef(j);
				
				if (x != null && y != null)
					C.setValue(i, j, x.innerProduct(y));
				else
					C.setValue(i, j, 0.0);
			}
		}
		
		return C;
	}
	
	/**
	 * Matrix-matrix product (A = AB), without using extra memory.
	 * 
	 * @param B The matrix to be multiplied to this matrix.
	 * @throws RuntimeException when dimensions disagree
	 */
	public void selfTimes(SparseMatrix B) {
		// original implementation
		if (N != (B.length())[0])
			throw new RuntimeException("Dimensions disagree");
		
		for (int i = 0; i < M; i++) {
			SparseVector tmp = new SparseVector(N);
			for (int j = 0; j < (B.length())[1]; j++) {
				SparseVector x = this.getRowRef(i);
				SparseVector y = B.getColRef(j);
				
				if (x != null && y != null)
					tmp.setValue(j, x.innerProduct(y));
				else
					tmp.setValue(j, 0.0);
			}
			
			for (int j = 0; j < (B.length())[1]; j++) {
				this.setValue(i, j, tmp.getValue(j));
			}
		}
	}

	/**
	 * Matrix-matrix sum (C = A + B)
	 * </br>���������������
	 * @param B The matrix to be added to this matrix.
	 * @throws RuntimeException when dimensions disagree
	 * @return The resulting matrix after summation.
	 */
	public SparseMatrix plus(SparseMatrix B) {
		SparseMatrix A = this;
		if (A.M != B.M || A.N != B.N)
			throw new RuntimeException("Dimensions disagree");
		
		SparseMatrix C = new SparseMatrix(M, N);
		for (int i = 0; i < M; i++) {
			C.rows[i] = A.rows[i].plus(B.rows[i]);
		}
		for (int j = 0; j < N; j++) {
			C.cols[j] = A.cols[j].plus(B.cols[j]);
		}
		
		return C;
	}
	
	/**
	 * Generate an identity matrix with the given size.</br>
	 * 	����һ��featureCount*featureCount��λ����
	 * @param n The size of requested identity matrix.
	 * @return An identity matrix with the size of n by n. 
	 */
	public static SparseMatrix makeIdentity(int n) {
		SparseMatrix m = new SparseMatrix(n, n);
		for (int i = 0; i < n; i++) {
			m.setValue(i, i, 1.0);
		}
		
		return m;
	}
	
	/**
	 * Calculate inverse matrix.
	 * </br>���㷽��������
	 * @throws RuntimeException when dimensions disagree.
	 * @return The inverse of current matrix.
	 */
	public SparseMatrix inverse() {
		if (this.M != this.N)
			throw new RuntimeException("Dimensions disagree");
		
		SparseMatrix original = this;
		SparseMatrix newMatrix = makeIdentity(this.M);
		
		int n = this.M;
		
		if (n == 1) {
			newMatrix.setValue(0, 0, 1 / original.getValue(0, 0));
			return newMatrix;
		}

		SparseMatrix b = new SparseMatrix(original);
		
		for (int i = 0; i < n; i++) {
			// find pivot:
			double mag = 0;
			int pivot = -1;

			for (int j = i; j < n; j++) {
				double mag2 = Math.abs(b.getValue(j, i));
				if (mag2 > mag) {
					mag = mag2;
					pivot = j;
				}
			}

			// no pivot (error):
			if (pivot == -1 || mag == 0) {
				return newMatrix;
			}

			// move pivot row into position:
			if (pivot != i) {
				double temp;
				for (int j = i; j < n; j++) {
					temp = b.getValue(i, j);
					b.setValue(i, j, b.getValue(pivot, j));
					b.setValue(pivot, j, temp);
				}

				for (int j = 0; j < n; j++) {
					temp = newMatrix.getValue(i, j);
					newMatrix.setValue(i, j, newMatrix.getValue(pivot, j));
					newMatrix.setValue(pivot, j, temp);
				}
			}

			// normalize pivot row:
			mag = b.getValue(i, i);
			for (int j = i; j < n; j ++) {
				b.setValue(i, j, b.getValue(i, j) / mag);
			}
			for (int j = 0; j < n; j ++) {
				newMatrix.setValue(i, j, newMatrix.getValue(i, j) / mag);
			}

			// eliminate pivot row component from other rows:
			for (int k = 0; k < n; k ++) {
				if (k == i)
					continue;
				
				double mag2 = b.getValue(k, i);

				for (int j = i; j < n; j ++) {
					b.setValue(k, j, b.getValue(k, j) - mag2 * b.getValue(i, j));
				}
				for (int j = 0; j < n; j ++) {
					newMatrix.setValue(k, j, newMatrix.getValue(k, j) - mag2 * newMatrix.getValue(i, j));
				}
			}
		}
		
		return newMatrix;
	}
	
	/**
	 * Calculate Cholesky decomposition of the matrix.
	 * </br>����þ���Ŀ���˹��ֽ����
	 * @throws RuntimeException when matrix is not square.
	 * @return The Cholesky decomposition result.
	 */
	public SparseMatrix cholesky() {
		if (this.M != this.N)
			throw new RuntimeException("Matrix is not square");
		
		SparseMatrix A = this;
		
		int n = A.M;
		SparseMatrix L = new SparseMatrix(n, n);

		for (int i = 0; i < n; i++)  {
			for (int j = 0; j <= i; j++) {
				double sum = 0.0;
				for (int k = 0; k < j; k++) {
					sum += L.getValue(i, k) * L.getValue(j, k);
				}
				if (i == j) {
					L.setValue(i, i, Math.sqrt(A.getValue(i, i) - sum));
				}
				else {
					L.setValue(i, j, 1.0 / L.getValue(j, j) * (A.getValue(i, j) - sum));
				}
			}
			if (Double.isNaN(L.getValue(i, i))) {
				//throw new RuntimeException("Matrix not positive definite: (" + i + ", " + i + ")");
				return null;
			}
		}
		
		return L.transpose();
	}
	
	/**
	 * Generate a covariance matrix of the current matrix.
	 * </br>��øþ����Э�������
	 * @return The covariance matrix of the current matrix.
	 */
	public SparseMatrix covariance() {
		int columnSize = this.N;
		SparseMatrix cov = new SparseMatrix(columnSize, columnSize);
		
		for (int i = 0; i < columnSize; i++) {
			for (int j = i; j < columnSize; j++) {
				SparseVector data1 = this.getCol(i);
				SparseVector data2 = this.getCol(j);
				double avg1 = data1.average();
				double avg2 = data2.average();
				
				double value = data1.sub(avg1).innerProduct(data2.sub(avg2)) / (data1.length()-1);
				cov.setValue(i, j, value);
				cov.setValue(j, i, value);
			}
		}
		
		return cov;
	}
	
	/*========================================
	 * Matrix operations (partial)
	 *========================================*/
	/**
	 * Scalar Multiplication only with indices in indexList.
	 * 
	 * @param alpha The scalar to be multiplied to this matrix.
	 * @param indexList The list of indices to be applied summation.
	 * @return The resulting matrix after scaling.
	 */
	public SparseMatrix partScale(double alpha, int[] indexList) {
		if (indexList != null) {
			for (int i : indexList) {
				for (int j : indexList) {
					this.setValue(i, j, this.getValue(i, j) * alpha);
				}
			}
		}
		
		return this;
	}
	
	/**
	 * Matrix summation (A = A + B) only with indices in indexList.
	 * 
	 * @param B The matrix to be added to this matrix.
	 * @param indexList The list of indices to be applied summation.
	 * @throws RuntimeException when dimensions disagree.
	 * @return The resulting matrix after summation.
	 */
	public SparseMatrix partPlus(SparseMatrix B, int[] indexList) {
		if (indexList != null) {
			if (this.M != B.M || this.N != B.N)
				throw new RuntimeException("Dimensions disagree");
			
			for (int i : indexList) {
				this.rows[i].partPlus(B.rows[i], indexList);
			}
			for (int j : indexList) {
				this.cols[j].partPlus(B.cols[j], indexList);
			}
		}
		
		return this;
	}
	
	/**
	 * Matrix subtraction (A = A - B) only with indices in indexList.
	 * 
	 * @param B The matrix to be subtracted from this matrix.
	 * @param indexList The list of indices to be applied subtraction.
	 * @throws RuntimeException when dimensions disagree.
	 * @return The resulting matrix after subtraction.
	 */
	public SparseMatrix partMinus(SparseMatrix B, int[] indexList) {
		if (indexList != null) {
			if (this.M != B.M || this.N != B.N)
				throw new RuntimeException("Dimensions disagree");
			
			for (int i : indexList) {
				this.rows[i].partMinus(B.rows[i], indexList);
			}
			for (int j : indexList) {
				this.cols[j].partMinus(B.cols[j], indexList);
			}
		}
		
		return this;
	}
	
	/**
	 * Matrix-vector product (b = Ax) only with indices in indexList.
	 * 
	 * @param x The vector to be multiplied to this matrix.
	 * @param indexList The list of indices to be applied multiplication.
	 * @return The resulting vector after multiplication.
	 */
	public SparseVector partTimes(SparseVector x, int[] indexList) {
		if (indexList == null)
			return x;
		
		SparseVector b = new SparseVector(M);
		
		for (int i : indexList) {
			b.setValue(i, this.rows[i].partInnerProduct(x, indexList));
		}
		
		return b;
	}
	
	/**
	 * Inverse of matrix only with indices in indexList.
	 * 
	 * @param indexList The list of indices to be applied multiplication.
	 * @throws RuntimeException when dimensions disagree.
	 * @return The resulting inverse matrix.
	 */
	public SparseMatrix partInverse(int[] indexList) {
		if (indexList == null)
			return this;
		
		if (this.M != this.N)
			throw new RuntimeException("Dimensions disagree");
		
		DenseMatrix dm = this.toDenseSubset(indexList);
		dm = dm.inverse();
		SparseMatrix newMatrix = new SparseMatrix(this.M, this.N);
		
		int x = 0;
		for (int i : indexList) {
			int y = 0;
			for (int j : indexList) {
				newMatrix.setValue(i, j, dm.getValue(x, y));
				y++;
			}
			x++;
		}
		
		return newMatrix;
	}
	
	/**
	 * Convert the matrix to a printable string.
	 * 
	 * @return The resulted string in the form of "(1, 2: 5.0) (2, 4: 4.5)"
	 */
	@Override
	public String toString() {
        String s = "";
        
        for (int i = 0; i < this.M; i++) {
        	SparseVector row = this.getRowRef(i);
        	for (int j : row.indexList()) {
        		s += "(" + i + ", " + j + ": " + this.getValue(i, j) + ") ";
        	}
        	s += "\r\n";
        }
        
        return s;
    }
}
