Changelog
---------

* **Pyflex 0.1.5** *Jul 11th 2016*
    * Make it work with the latest versions of ObsPy, flake8, and matplotlib.

* **Pyflex 0.1.4** *Sep 30th 2015*
    * Minor change to adapt tests to the latest ObsPy version.
    * Dropped official support for Python 2.6. It still works with it for now but I don't plan on further supporting it.

* **Pyflex 0.1.3** *Dez 10th 2014*
    * Rejecting windows with very early start or very late end times. Greatly speeds up the algorithm.
    * New config parameter ``max_time_before_first_arrival``.
    * New optional time normalized energy signal to noise ratio. More resilient to random wiggles before the first arrival.
    * New config parameter ``window_signal_to_noise_type``.

* **Pyflex 0.1.2** *Nov 17th 2014*
    * Support for (de)serializing windows to and from JSON.
    * Much faster STA/LTA computation.

* **Pyflex 0.1.1** *Nov 11th 2014*
    * Some bugfixes regarding units.

* **Pyflex 0.1.0** *Nov 9th 2014*
    * Initial Pyflex release.
