"""Tests for filter.py"""
import unittest
from geojson import Feature, Polygon, LineString

from label_maker.filter import create_filter, _compile, _compile_property_reference, \
     _compile_comparison_op, _compile_logical_op, _compile_in_op, _compile_has_op, \
     _compile_negation, _stringify

line_geometry = LineString([(0, 0), (1, 1)])
polygon_geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

class TestCompiledFilters(unittest.TestCase):
    """Tests for compiled filter functions"""

    def test_comparison(self):
        """Test comparison filter function"""
        ff = create_filter(['==', 'a', 5])
        passing = Feature(geometry=line_geometry, properties=dict(a=5))
        failing = Feature(geometry=line_geometry, properties=dict(a=4))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

    def test_any(self):
        """Test any filter function"""
        ff = create_filter(['any', ['==', 'a', 5], ['==', 'b', 3]])
        passing1 = Feature(geometry=line_geometry, properties=dict(a=5))
        passing2 = Feature(geometry=line_geometry, properties=dict(b=3))
        passing3 = Feature(geometry=line_geometry, properties=dict(a=5, b=3))
        failing1 = Feature(geometry=line_geometry, properties=dict(a=4))
        failing2 = Feature(geometry=line_geometry, properties=dict(b=5))
        self.assertTrue(ff(passing1))
        self.assertTrue(ff(passing2))
        self.assertTrue(ff(passing3))
        self.assertFalse(ff(failing1))
        self.assertFalse(ff(failing2))

    def test_all(self):
        """Test all filter function"""
        ff = create_filter(['all', ['==', 'a', 5], ['==', 'b', 3]])
        passing = Feature(geometry=line_geometry, properties=dict(a=5, b=3))
        failing1 = Feature(geometry=line_geometry, properties=dict(a=5))
        failing2 = Feature(geometry=line_geometry, properties=dict(b=3))
        failing3 = Feature(geometry=line_geometry, properties=dict(b=1))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing1))
        self.assertFalse(ff(failing2))
        self.assertFalse(ff(failing3))

    def test_none(self):
        """Test none filter function"""
        ff = create_filter(['none', ['==', 'a', 5], ['==', 'b', 3]])
        passing = Feature(geometry=line_geometry, properties=dict(b=1))
        failing1 = Feature(geometry=line_geometry, properties=dict(a=5, b=3))
        failing2 = Feature(geometry=line_geometry, properties=dict(a=5))
        failing3 = Feature(geometry=line_geometry, properties=dict(b=3))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing1))
        self.assertFalse(ff(failing2))
        self.assertFalse(ff(failing3))

    def test_in(self):
        """Test in filter function"""
        ff = create_filter(['in', 'a', 1, 2])
        passing = Feature(geometry=line_geometry, properties=dict(a=1))
        failing = Feature(geometry=line_geometry, properties=dict(a=3))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

    def test_not_in(self):
        """Test !in filter function"""
        ff = create_filter(['!in', 'a', 1, 2])
        passing = Feature(geometry=line_geometry, properties=dict(a=3))
        failing = Feature(geometry=line_geometry, properties=dict(a=1))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

    def test_has(self):
        """Test has filter function"""
        ff = create_filter(['has', 'a'])
        passing = Feature(geometry=line_geometry, properties=dict(a=1))
        failing = Feature(geometry=line_geometry, properties=dict(b=3))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

    def test_not_has(self):
        """Test !has filter function"""
        ff = create_filter(['!has', 'a'])
        passing = Feature(geometry=line_geometry, properties=dict(b=3))
        failing = Feature(geometry=line_geometry, properties=dict(a=1))
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

    def test_geometry_comparison(self):
        """Test $type specific filters for comparison"""
        ff = create_filter(['==', '$type', 'Polygon'])
        # print(_compile(['==', '$type', 'Polygon']))
        passing = Feature(geometry=polygon_geometry)
        failing = Feature(geometry=line_geometry)
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

    def test_geometry_in(self):
        """Test $type specific filters for inclusion"""
        ff = create_filter(['in', '$type', 'Point', 'Polygon'])
        # print(_compile(['in', '$type', 'Point', 'Polygon']))
        passing = Feature(geometry=polygon_geometry)
        failing = Feature(geometry=line_geometry)
        self.assertTrue(ff(passing))
        self.assertFalse(ff(failing))

class TestFilter(unittest.TestCase):
    """Tests for private filter.py functions"""

    def test_compile(self):
        """Test private function _compile"""
        self.assertEqual(_compile(['==', 'a', 5]), 'p.get("a")==5')
        self.assertEqual(_compile(['any', ['==', 'a', 5], ['==', 'b', 3]]), 'p.get("a")==5 or p.get("b")==3')
        self.assertEqual(_compile(['all', ['==', 'a', 5], ['==', 'b', 3]]), 'p.get("a")==5 and p.get("b")==3')
        self.assertEqual(_compile(['none', ['==', 'a', 5], ['==', 'b', 3]]), 'not (p.get("a")==5 or p.get("b")==3)')
        self.assertEqual(_compile(['in', 'a', 1, 2]), 'p.get("a") in [1, 2]')
        self.assertEqual(_compile(['!in', 'a', 1, 2]), 'not (p.get("a") in [1, 2])')
        self.assertEqual(_compile(['has', 'a']), '"a" in p')
        self.assertEqual(_compile(['!has', 'a']), 'not ("a" in p)')

    def test_compile_property_reference(self):
        """Test private function _compile_property_reference"""
        self.assertEqual(_compile_property_reference('$type'), 'f.get("geometry").get("type")')
        self.assertEqual(_compile_property_reference('$id'), 'f.get("id")')
        self.assertEqual(_compile_property_reference('a'), 'p.get("a")')

    def test_compile_comparison_op(self):
        """Test private function _compile_comparison_op"""
        self.assertEqual(_compile_comparison_op('a', 5, '=='), 'p.get("a")==5')
        self.assertEqual(_compile_comparison_op('$type', 'Polygon', '=='), 'f.get("geometry").get("type")=="Polygon"')

    def test_compile_logical_op(self):
        """Test private function _compile_logical_op"""
        self.assertEqual(_compile_logical_op([['==', 'a', 5], ['==', 'b', 3]], ' and '), 'p.get("a")==5 and p.get("b")==3')

    def test_compile_in_op(self):
        """Test private function _compile_in_op"""
        self.assertEqual(_compile_in_op('a', ['a', 'b']), 'p.get("a") in [\'a\', \'b\']')
        self.assertEqual(_compile_in_op('$type', ['Point', 'Polygon']), 'f.get("geometry").get("type") in [\'Point\', \'Polygon\']')

    def test_compile_has_op(self):
        """Test private function _compile_has_op"""
        self.assertEqual(_compile_has_op('a'), '"a" in p')
        self.assertEqual(_compile_has_op('$id'), '"id" in f')

    def test_compile_negation(self):
        """Test private function _compile_negation"""
        self.assertEqual(_compile_negation('a'), 'not (a)')

    def test_stringify(self):
        """Test private function _stringify"""
        self.assertEqual(_stringify(5), '5')
        self.assertEqual(_stringify('a'), '"a"')

if __name__ == '__main__':
    unittest.main()
