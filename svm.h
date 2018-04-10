#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 322

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

/**
* `struct svm_node` 用来存储单一向量中的单个特征，
* 例如： 向量  `x1={ 0.002, 0.345, 4, 5.677};`
* 那么用 `struct svm_node` 来存储时就使用一个包含
* 5个svm_node的数组来存储此4维向量，内存映象如下：
* |   1   |   2   |   3   |   4   |  －1  |
* |  ---  |  ---  |  ---  |  ---  |  ---  |
* | 0.002 | 0.345 | 4.000 | 5.677 |  空   |
*
* 其中如果 value 为 0.00,该特征将不会被存储，如下，其中(特征 3)被跳过：
* |   1   |   2   |   4   |   5   |  －1  |
* |  ---  |  ---  |  ---  |  ---  |  ---  |
* | 0.002 | 0.345 | 4.000 | 5.677 |  空   |
*
* 0.00 不保留的好处在于，做点乘的时候，可以加快计算速度，对于稀疏矩阵，
* 更能充分体现这种数据结构的优势（但做归一化时，操作就比较麻烦了）
*/
struct svm_node
{
	int index;
	double value;
};

/**
* `struct svm_problem`存储本次参加运算的所有样本（数据集），及其所属类别
*/
struct svm_problem
{
	int l;  // 记录样本总数 
	double *y; // 指向样本所属类别的数组
	struct svm_node **x; // 指向一个存储内容为`svm_node`指针的数组
};

/* svm_type */
enum { 
	/** C-Support Vector Classification. n-class classification (n \f$\geq\f$ 2), allows
	* imperfect separation of classes with penalty multiplier C for outliers. 
	*/
	C_SVC,

	/** \f$\nu\f$-Support Vector Classification. n-class classification with possible
	* imperfect separation. Parameter \f$\nu\f$ (in the range 0..1, the larger the value, the smoother
	* the decision boundary) is used instead of C. 
	*/
	NU_SVC, 

	/** Distribution Estimation (One-class %SVM). All the training data are from
	* the same class, %SVM builds a boundary that separates the class from the rest of the feature
	* space. 
	*/
	ONE_CLASS, 

	/** \f$\epsilon\f$-Support Vector Regression. The distance between feature vectors
	* from the training set and the fitting hyper-plane must be less than p. For outliers the
	* penalty multiplier C is used. 
	*/
	EPSILON_SVR, 

	/** \f$\nu\f$-Support Vector Regression. \f$\nu\f$ is used instead of p.
	* See @cite LibSVM for details. 
	*/
	NU_SVR 
};

/* kernel_type */
enum { 
	/** Linear kernel. 
	* 没有做特征空间映射，线性判别（或回归）是在原始特征空间中完成，参数少，速度快
	* \f$K(x_i, x_j) = x_i^T x_j\f$. 
	*/  // 公式 1-1
	LINEAR, 

	/** Polynomial kernel. 
	* \f$K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0\f$. 
	*/ // 公式 1-2
	POLY, 

	/** Radial basis function (RBF).
	* \f$K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0\f$. 
	*/ // 公式 1-3
	RBF,  

	/** Sigmoid kernel.
	* \f$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0)\f$. 
	*/ // 公式 1-4
	SIGMOID,

	/**/
	PRECOMPUTED 
}; 

/* 参数设置参看 `kernel_type` 中的 '公式 1-1 ~ 1-4 ' */
struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */  // 制定训练所需要的内存，默认是40M
	double eps;	/* stopping criteria */   // 误差限
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */  // 惩罚因子，越大，训练的模型耗的时间越多
	int nr_weight;		/* for C_SVC */  // 权重的数目，目前在实例代码中只有两个值，一个是默认0，另外一个是`svm_binary_svc_probability`函数中使用数值2
	int *weight_label;	/* for C_SVC */  // 权重，元素个数由nr_weight决定
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */  // 指明训练过程是否使用压缩
	int probability; /* do probability estimates */  // 指明是否要做概率估计
};

//
// svm_model
// 
/* `svm_model`用于保存训练后的训练模型，并保留原来的训练参数 */
struct svm_model
{
	struct svm_parameter param;	/* parameter */  // 训练参数 
	int nr_class;		/* number of classes, = 2 in regression/one class svm */ // 类别数 
	int l;			/* total #SV */  // 支持向量数
	struct svm_node **SV;		/* SVs (SV[l]) */ // 保存支持向量的指针
												  // 至于支持向量的内容，如果是从文件中读取，内容会
												  // 额外保留；如果是直接训练得来，则保留在原来的训练集中
												  // 如果训练完成后需要预报，原来的训练集内存不可以释放

	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */  // 相当于判别函数中的alpha
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */  // 相当于判别函数中的b
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */

	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				       /* 0 if svm_model is created by svm_train */  // 参见上述`struct svm_node **SV;`注释
};

/* 训练数据 */
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
/* 用SVM做交叉验证 */
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
/* 将训练好的模型保存到文件 */
int svm_save_model(const char *model_file_name, const struct svm_model *model);
/* 加载训练好的模型 */
struct svm_model *svm_load_model(const char *model_file_name);

/**/
int svm_get_svm_type(const struct svm_model *model);
/* 得到数据集的类别数（必须经过训练得到模型后才可以用）*/
int svm_get_nr_class(const struct svm_model *model);
/* 得到数据集的类别标号（必须经过训练得到模型后才可以用）*/
void svm_get_labels(const struct svm_model *model, int *label);
/**/
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
/**/
int svm_get_nr_sv(const struct svm_model *model);
/**/
double svm_get_svr_probability(const struct svm_model *model);
/* 用训练好的模型预报样本的值，输出结果保留到数组中（并非接口函数）*/
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
/* 预报某一样本的值 */
double svm_predict(const struct svm_model *model, const struct svm_node *x);
/**/
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

/* 释放资源 */
void svm_free_model_content(struct svm_model *model_ptr);
/**/
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
/**/
void svm_destroy_param(struct svm_parameter *param);

/* 检查输入的参数，保证之后的训练能正常进行 */
const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
/**/
int svm_check_probability_model(const struct svm_model *model);

/**/
void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
