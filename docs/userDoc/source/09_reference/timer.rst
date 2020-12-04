.. include:: ../include/global.inc

timer
*********

It is often required to repeatedly call a specific function or method with a certain time interval.
This can be done using the class *timer*. See the example *timerExample.py* in the *demo* folder
for what the timer can be used. The method or function can either be called once after a certain
interval (single shot timer) or repeatedly with a given interval. In both cases, the variable that
references the timer instance must always exist. Once it is destroyed, the timer is automatically stopped
and deleted. Use the :ref:`timer manager <gui-timermanager>` (menu *scripts >> timer manager*) to
see a list of all active timer instances. This dialog also allows stopping or restarting these timers.

.. currentmodule:: itom

.. autoclass:: itom.timer
    :member-order: groupwise
    :members:
    :undoc-members: