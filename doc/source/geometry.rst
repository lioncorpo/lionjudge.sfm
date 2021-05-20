.. Doc on geometric models and coordinate systems

Geometric Models
================


Coordinate Systems
------------------

Normalized Image Coordinates
````````````````````````````

The 2d position of a point in images is stored in what we will call *normalized image coordinates*.  The origin is in the middle of the image.  The x coordinate grows to the right and y grows downwards.  The larger dimension of the image is 1.

This means, for example, that all the pixels in an image with aspect ratio 4:3 will be contained in the intervals ``[-0.5, 0.5]`` and ``[3/4 * (-0.5), 3/4 * 0.5]`` for the X and Y axis respectively.

::

     +-----------------------------+
     |                             |
     |                             |
     |                             |
     |              + ------------->
     |              | (0, 0)       | (0.5, 0)
     |              |              |
     |              |              |
     +-----------------------------+
                    |
                    v
                     (0, 0.5)

Normalized coordinates are independent of the resolution of the image and give better numerical stability for some multi-view geometry algorithms than pixel coordinates.


Pixel Coordinates
`````````````````

Many OpenCV functions that work with images use *pixel coordinates*.  In that reference frame, the origin is at the center of the top-left pixel, x grow by one for every pixel to the right and y grows by one for every pixel downwards.  The bottom-right pixel is therefore at ``(width - 1, height - 1)``.

The transformation from normalised image coordinates to pixel coordinates is

.. math::
   H = \begin{pmatrix}
            \max(w, h) & 0 & \frac{w-1}{2} \\
            0 & \max(w, h) & \frac{h-1}{2} \\
            0 & 0 & 1
        \end{pmatrix},

and its inverse

.. math::
   H^{-1} = \begin{pmatrix}
            1 & 0 & -\frac{w-1}{2} \\
            0 & 1 & -\frac{h-1}{2} \\
            0 & 0 & \max(w, h)
        \end{pmatrix},

where :math:`w` and :math:`h` being the width and height of the image.

World Coordinates
`````````````````
The position of the reconstructed 3D points is stored in *world coordinates*.  In general, this is an arbitrary euclidean reference frame.

When GPS data is available, a topocentric reference frame is used for the world coordinates reference.  This is a reference frame that with the origin somewhere near the ground, the X axis pointing to the east, the Y axis pointing to the north and the Z axis pointing to the zenith.  The latitude, longitude, and altitude of the origin are stored in the ``reference_lla.json`` file.

When GPS data is not available, the reconstruction process makes its best to rotate the world reference frame so that the vertical direction is Z and the ground is near the `z = 0` plane.  It does so by assuming that the images are taken from similar altitudes and that the up vector of the images corresponds to the up vector of the world.


Camera Coordinates
``````````````````
The *camera coordinate* reference frame has the origin at the camera's optical center, the X axis is pointing to the right of the camera the Y axis is pointing down and the Z axis is pointing to the front.  A point in front of the camera has positive Z camera coordinate.

The pose of a camera is determined by the rotation and translation that converts world coordinates to camera coordinates.


Camera Models
-------------
The camera models deal with the projection of 3D points expressed in *camera coordinates* ``x, y, z`` into points ``u, v`` in *normalized image coordinates*.

Perspective Camera
``````````````````
.. math::
    \begin{array}{l}
    x_n = \frac{x}{z} \\
    y_n = \frac{y}{z} \\
    r^2 = x_n^2 + y_n^2 \\
    d = 1 + k_1 r^2 + k_2 r^4 \\
    u = f\ d\ x_n \\
    v = f\ d\ y_n
    \end{array}

Fisheye Camera
``````````````````
.. math::
    \begin{array}{l}
    r^2 = x^2 + y^2 \\
    \theta = \arctan(r / z) \\
    d = 1 +  k_1 \theta^2+  k_2 \theta^4 \\
    u = f\ d\ \theta\ \frac{x}{r} \\
    v = f\ d\ \theta\ \frac{y}{r}
    \end{array}

Spherical Camera
``````````````````
.. math::
    \begin{array}{l}
    \mathrm{lon} = \arctan\left(\frac{x}{z}\right) \\
    \mathrm{lat} = \arctan\left(\frac{-y}{\sqrt{x^2 + z^2}}\right) \\
    u = \frac{\mathrm{lon}}{2 \pi} \\
    v = -\frac{\mathrm{lat}}{2 \pi}
    \end{array}

