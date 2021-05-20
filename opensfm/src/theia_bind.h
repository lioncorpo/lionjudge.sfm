#pragma once

#include "types.h"

namespace csfm {

py::object match_using_cascade_hashing(pyarray_f features1,
                                       pyarray_f features2,
                                       float lowes_ratio);

}
