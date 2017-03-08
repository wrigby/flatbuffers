# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Accessor classes that provide a Pythonic interface to vectors """


class VectorAccessor(object):
    """ Base class that type-specific vector accessors inherit from """

    __slots__ = ["_table", "_offset", "_length", "_stride"]

    def __init__(self, table, offset, stride):
        """ Creates the base vector
        
        Parameters:
            table  - The table this vector is part of (a flatbuffers.Table object)
            offset - The offset in the table to the vector we're accessing
            stride - The distance between each element of the vector in bytes
        """

        self._table = table
        self._offset = offset
        self._length = table.VectorLen(offset)
        self._stride = stride

    def __iter__(self):
        """ Return an iterator for this vector """
        return VectorIterator(self)

    def __getitem__(self, index):
        """ Gets the n'th item in this vector """
        if not isinstance(index, int):
            raise TypeError("vector indices must be integers, not %s" %
                            type(index).__name__)

        if index < 0 or index >= self._length:
            raise IndexError("vector index out of range")

        a = self._table.Vector(self._offset) + index * self._stride
        return self._unpack(a)

    def __len__(self):
        """ Returns the length of this vector """
        return self._length

    def _unpack(self, offset):
        """ To be implemented by child classes """
        raise NotImplementedError


class StructVectorAccessor(VectorAccessor):
    """ Accessor for vectors of structs """

    __slots__ = ["_struct_class"]

    def __init__(self, table, offset, stride, struct_class):
        super(StructVectorAccessor, self).__init__(table, offset, stride)
        self._struct_class = struct_class

    def _unpack(self, offset):
        item = self._struct_class()
        item.Init(self._table.Bytes, offset)
        return item


class SimpleVectorAccessor(VectorAccessor):
    """ Handles vectors of base (number and boolean) types """

    __slots__ = ["_flags"]

    def __init__(self, table, offset, flags):
        super(SimpleVectorAccessor, self).__init__(table, offset, flags.bytewidth)
        self._flags = flags

    def _unpack(self, offset):
        return self._table.Get(self._flags, offset)


class StringVectorAccessor(VectorAccessor):
    """ Handles vectors of strings """

    def __init__(self, table, offset, stride):
        super(SimpleVectorAccessor, self).__init__(table, offset, stride)

    def _unpack(self, offset):
        return self._table.String(offset)


class VectorIterator(object):
    """ Iterator used to iterator over a VectorAccessor """

    __slots__ = ["_vec", "_next_index", "_length"]

    def __init__(self, vector_accessor):
        self._vec = vector_accessor
        self._next_index = 0
        self._length = len(vector)

    def __iter__(self):
        return self

    def next(self):
        """ Returns the next item in the vector """
        if self._next_index >= self._length:
            raise StopIteration

        item = self._vec[self._next_index]
        self._next_index += 1
        return item
