#ifndef IONICMODEL
#define IONICMODEL

double inline NV_Ith_S(double* y, unsigned i)
{
    return y[i];
}

double inline phi_f_from_lmbda_yinf(double y, double lmbda, double yinf, double dt)
{
    return ((exp(dt*lmbda)-1.)/dt)*(y-yinf);
}

double inline phi_f_from_tau_yinf(double y, double tau, double yinf, double dt)
{
    return ((exp(-dt/tau)-1.)/dt)*(y-yinf);
}

void get_raw_data(const py::list& array_list, double** array_ptrs, size_t& N, size_t& n_dofs)
{
    N = array_list.size();
    unsigned i = 0;
    for( py::handle array: array_list)
    {
        py::array_t<double> casted_array = py::cast<py::array>(array);
        auto requestCastedArray = casted_array.request();
        n_dofs = requestCastedArray.shape[0];
        array_ptrs[i] = (double*) requestCastedArray.ptr;
        i++;
    }
};

void assign(py::list l, std::initializer_list<int> a)
{
    for(auto a_el:a)
        l.append(a_el);
};

class IonicModel
{
public:
    IonicModel(const double scale_);

    py::list f_nonstiff_args;
    py::list f_stiff_args;
    py::list f_expl_args;
    py::list f_exp_args;
    py::list f_nonstiff_indeces;
    py::list f_stiff_indeces;
    py::list f_expl_indeces;
    py::list f_exp_indeces;
    py::list get_f_nonstiff_args(){return f_nonstiff_args;};
    py::list get_f_stiff_args(){return f_stiff_args;};    
    py::list get_f_expl_args(){return f_expl_args;};    
    py::list get_f_exp_args(){return f_exp_args;};    
    py::list get_f_nonstiff_indeces(){return f_nonstiff_indeces;};
    py::list get_f_stiff_indeces(){return f_stiff_indeces;};    
    py::list get_f_expl_indeces(){return f_expl_indeces;};    
    py::list get_f_exp_indeces(){return f_exp_indeces;};    
    size_t get_size(){return size;};

protected:
    double scale;
    size_t size;
};

IonicModel::IonicModel(const double scale_)
{
    scale = scale_;
}

#endif