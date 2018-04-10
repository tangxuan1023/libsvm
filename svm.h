#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 322

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

/**
* `struct svm_node` �����洢��һ�����еĵ���������
* ���磺 ����  `x1={ 0.002, 0.345, 4, 5.677};`
* ��ô�� `struct svm_node` ���洢ʱ��ʹ��һ������
* 5��svm_node���������洢��4ά�������ڴ�ӳ�����£�
* |   1   |   2   |   3   |   4   |  ��1  |
* |  ---  |  ---  |  ---  |  ---  |  ---  |
* | 0.002 | 0.345 | 4.000 | 5.677 |  ��   |
*
* ������� value Ϊ 0.00,�����������ᱻ�洢�����£�����(���� 3)��������
* |   1   |   2   |   4   |   5   |  ��1  |
* |  ---  |  ---  |  ---  |  ---  |  ---  |
* | 0.002 | 0.345 | 4.000 | 5.677 |  ��   |
*
* 0.00 �������ĺô����ڣ�����˵�ʱ�򣬿��Լӿ�����ٶȣ�����ϡ�����
* ���ܳ�������������ݽṹ�����ƣ�������һ��ʱ�������ͱȽ��鷳�ˣ�
*/
struct svm_node
{
	int index;
	double value;
};

/**
* `struct svm_problem`�洢���βμ�������������������ݼ����������������
*/
struct svm_problem
{
	int l;  // ��¼�������� 
	double *y; // ָ������������������
	struct svm_node **x; // ָ��һ���洢����Ϊ`svm_node`ָ�������
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
	* û���������ռ�ӳ�䣬�����б𣨻�ع飩����ԭʼ�����ռ�����ɣ������٣��ٶȿ�
	* \f$K(x_i, x_j) = x_i^T x_j\f$. 
	*/  // ��ʽ 1-1
	LINEAR, 

	/** Polynomial kernel. 
	* \f$K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0\f$. 
	*/ // ��ʽ 1-2
	POLY, 

	/** Radial basis function (RBF).
	* \f$K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0\f$. 
	*/ // ��ʽ 1-3
	RBF,  

	/** Sigmoid kernel.
	* \f$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0)\f$. 
	*/ // ��ʽ 1-4
	SIGMOID,

	/**/
	PRECOMPUTED 
}; 

/* �������òο� `kernel_type` �е� '��ʽ 1-1 ~ 1-4 ' */
struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */  // �ƶ�ѵ������Ҫ���ڴ棬Ĭ����40M
	double eps;	/* stopping criteria */   // �����
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */  // �ͷ����ӣ�Խ��ѵ����ģ�ͺĵ�ʱ��Խ��
	int nr_weight;		/* for C_SVC */  // Ȩ�ص���Ŀ��Ŀǰ��ʵ��������ֻ������ֵ��һ����Ĭ��0������һ����`svm_binary_svc_probability`������ʹ����ֵ2
	int *weight_label;	/* for C_SVC */  // Ȩ�أ�Ԫ�ظ�����nr_weight����
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */  // ָ��ѵ�������Ƿ�ʹ��ѹ��
	int probability; /* do probability estimates */  // ָ���Ƿ�Ҫ�����ʹ���
};

//
// svm_model
// 
/* `svm_model`���ڱ���ѵ�����ѵ��ģ�ͣ�������ԭ����ѵ������ */
struct svm_model
{
	struct svm_parameter param;	/* parameter */  // ѵ������ 
	int nr_class;		/* number of classes, = 2 in regression/one class svm */ // ����� 
	int l;			/* total #SV */  // ֧��������
	struct svm_node **SV;		/* SVs (SV[l]) */ // ����֧��������ָ��
												  // ����֧�����������ݣ�����Ǵ��ļ��ж�ȡ�����ݻ�
												  // ���Ᵽ���������ֱ��ѵ��������������ԭ����ѵ������
												  // ���ѵ����ɺ���ҪԤ����ԭ����ѵ�����ڴ治�����ͷ�

	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */  // �൱���б����е�alpha
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */  // �൱���б����е�b
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */

	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				       /* 0 if svm_model is created by svm_train */  // �μ�����`struct svm_node **SV;`ע��
};

/* ѵ������ */
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
/* ��SVM��������֤ */
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
/* ��ѵ���õ�ģ�ͱ��浽�ļ� */
int svm_save_model(const char *model_file_name, const struct svm_model *model);
/* ����ѵ���õ�ģ�� */
struct svm_model *svm_load_model(const char *model_file_name);

/**/
int svm_get_svm_type(const struct svm_model *model);
/* �õ����ݼ�������������뾭��ѵ���õ�ģ�ͺ�ſ����ã�*/
int svm_get_nr_class(const struct svm_model *model);
/* �õ����ݼ�������ţ����뾭��ѵ���õ�ģ�ͺ�ſ����ã�*/
void svm_get_labels(const struct svm_model *model, int *label);
/**/
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
/**/
int svm_get_nr_sv(const struct svm_model *model);
/**/
double svm_get_svr_probability(const struct svm_model *model);
/* ��ѵ���õ�ģ��Ԥ��������ֵ�������������������У����ǽӿں�����*/
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
/* Ԥ��ĳһ������ֵ */
double svm_predict(const struct svm_model *model, const struct svm_node *x);
/**/
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

/* �ͷ���Դ */
void svm_free_model_content(struct svm_model *model_ptr);
/**/
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
/**/
void svm_destroy_param(struct svm_parameter *param);

/* �������Ĳ�������֤֮���ѵ������������ */
const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
/**/
int svm_check_probability_model(const struct svm_model *model);

/**/
void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
