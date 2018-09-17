Contributing
============

A list of issues and ongoing work is available on the Label Maker `issues page <https://github.com/developmentseed/label-maker/issues>`_. If you want to contribute code, the best way is to coordinate with the core developers via an issue or pull request conversation.

Development installation
^^^^^^^^^^^^^^^^^^^^^^^^
Fork Label Maker into your Github account. Then, clone the repo and install it locally with pip as follows:

.. code-block:: bash

	$ git clone git@github.com:your_user_name/label-maker.git
	$ cd  label-maker
	$ pip install -e .

Testing
^^^^^^^
Label Maker runs tests using ``unittest``. You can find unit tests at ``tests/unit`` and integration tests at ``tests/integration``.

Run a single test with:

.. code-block:: bash

	python -m unittest test/unit/test_validate.py

or an entire folder using:

.. code-block:: bash

	python -m unittest discover -v -s test/unit

More details on using ``unittest`` are `here <https://docs.python.org/3/library/unittest.html>`_.