"""
PyInstaller Extractor v2.0 (Supports pyinstaller 6.16.0, 6.15.0, 6.14.2, 6.14.1, 6.14.0, 6.13.0, 6.12.0, 6.11.1, 6.11.0, 6.10.0, 6.9.0, 6.8.0, 6.7.0, 6.6.0, 6.5.0, 6.4.0, 6.3.0, 6.2.0, 6.1.0, 6.0.0, 5.13.2, 5.13.1, 5.13.0, 5.12.0, 5.11.0, 5.10.1, 5.10.0, 5.9.0, 5.8.0, 5.7.0, 5.6.2, 5.6.1, 5.6, 5.5, 5.4.1, 5.4, 5.3, 5.2, 5.1, 5.0.1, 5.0, 4.10, 4.9, 4.8, 4.7, 4.6, 4.5.1, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.1, 2.0)
Author : Extreme Coders
E-mail : extremecoders(at)hotmail(dot)com
Web    : https://0xec.blogspot.com
Date   : 26-March-2020
Url    : https://github.com/extremecoders-re/pyinstxtractor
"""

from __future__ import print_function
import os
import struct
import marshal
import zlib
import sys
from uuid import uuid4 as uniquename


class CTOCEntry:
    def __init__(self, position, cmprsdDataSize, uncmprsdDataSize, cmprsFlag, typeCmprsData, name):
        self.position = position
        self.cmprsdDataSize = cmprsdDataSize
        self.uncmprsdDataSize = uncmprsdDataSize
        self.cmprsFlag = cmprsFlag
        self.typeCmprsData = typeCmprsData
        self.name = name


class PyInstArchive:
    PYINST20_COOKIE_SIZE = 24
    PYINST21_COOKIE_SIZE = 24 + 64
    MAGIC = b'MEI\014\013\012\013\016'

    def __init__(self, path):
        self.filePath = path
        self.pycMagic = b'\0' * 4
        self.barePycList = []

    def open(self):
        try:
            self.fPtr = open(self.filePath, 'rb')
            self.fileSize = os.stat(self.filePath).st_size
        except:
            print('[!] Error: Could not open {0}'.format(self.filePath))
            return False
        return True

    def close(self):
        try:
            self.fPtr.close()
        except:
            pass

    def checkFile(self):
        print('[+] Processing {0}'.format(self.filePath))
        searchChunkSize = 8192
        endPos = self.fileSize
        self.cookiePos = -1

        if endPos < len(self.MAGIC):
            print('[!] Error : File is too short or truncated')
            return False

        while True:
            startPos = endPos - searchChunkSize if endPos >= searchChunkSize else 0
            chunkSize = endPos - startPos
            if chunkSize < len(self.MAGIC):
                break
            self.fPtr.seek(startPos, os.SEEK_SET)
            data = self.fPtr.read(chunkSize)
            offs = data.rfind(self.MAGIC)
            if offs != -1:
                self.cookiePos = startPos + offs
                break
            endPos = startPos + len(self.MAGIC) - 1
            if startPos == 0:
                break

        if self.cookiePos == -1:
            print('[!] Error : Missing cookie, unsupported pyinstaller version or not a pyinstaller archive')
            return False

        self.fPtr.seek(self.cookiePos + self.PYINST20_COOKIE_SIZE, os.SEEK_SET)
        if b'python' in self.fPtr.read(64).lower():
            print('[+] Pyinstaller version: 2.1+')
            self.pyinstVer = 21
        else:
            self.pyinstVer = 20
            print('[+] Pyinstaller version: 2.0')
        return True

    def getCArchiveInfo(self):
        try:
            if self.pyinstVer == 20:
                self.fPtr.seek(self.cookiePos, os.SEEK_SET)
                (magic, lengthofPackage, toc, tocLen, pyver) = \
                struct.unpack('!8siiii', self.fPtr.read(self.PYINST20_COOKIE_SIZE))
            elif self.pyinstVer == 21:
                self.fPtr.seek(self.cookiePos, os.SEEK_SET)
                (magic, lengthofPackage, toc, tocLen, pyver, pylibname) = \
                struct.unpack('!8sIIii64s', self.fPtr.read(self.PYINST21_COOKIE_SIZE))
        except:
            print('[!] Error : The file is not a pyinstaller archive')
            return False

        self.pymaj, self.pymin = (pyver//100, pyver%100) if pyver >= 100 else (pyver//10, pyver%10)
        print('[+] Python version: {0}.{1}'.format(self.pymaj, self.pymin))

        tailBytes = self.fileSize - self.cookiePos - (self.PYINST20_COOKIE_SIZE if self.pyinstVer == 20 else self.PYINST21_COOKIE_SIZE)
        self.overlaySize = lengthofPackage + tailBytes
        self.overlayPos = self.fileSize - self.overlaySize
        self.tableOfContentsPos = self.overlayPos + toc
        self.tableOfContentsSize = tocLen
        print('[+] Length of package: {0} bytes'.format(lengthofPackage))
        return True

    def parseTOC(self):
        self.fPtr.seek(self.tableOfContentsPos, os.SEEK_SET)
        self.tocList = []
        parsedLen = 0

        while parsedLen < self.tableOfContentsSize:
            (entrySize, ) = struct.unpack('!i', self.fPtr.read(4))
            nameLen = struct.calcsize('!iIIIBc')
            (entryPos, cmprsdDataSize, uncmprsdDataSize, cmprsFlag, typeCmprsData, name) = \
            struct.unpack('!IIIBc{0}s'.format(entrySize - nameLen), self.fPtr.read(entrySize - 4))

            try:
                name = name.decode("utf-8").rstrip("\0")
            except UnicodeDecodeError:
                newName = str(uniquename())
                print('[!] Warning: File name {0} contains invalid bytes. Using random name {1}'.format(name, newName))
                name = newName

            if name.startswith("/"):
                name = name.lstrip("/")

            if len(name) == 0:
                name = str(uniquename())
                print('[!] Warning: Found an unamed file in CArchive. Using random name {0}'.format(name))

            self.tocList.append(CTOCEntry(
                self.overlayPos + entryPos,
                cmprsdDataSize,
                uncmprsdDataSize,
                cmprsFlag,
                typeCmprsData,
                name
            ))
            parsedLen += entrySize
        print('[+] Found {0} files in CArchive'.format(len(self.tocList)))

    def _writeRawData(self, filepath, data):
        nm = filepath.replace('\\', os.path.sep).replace('/', os.path.sep).replace('..', '__')
        nmDir = os.path.dirname(nm)
        if nmDir != '' and not os.path.exists(nmDir):
            os.makedirs(nmDir)
        with open(nm, 'wb') as f:
            f.write(data)

    def extractFiles(self):
        print('[+] Beginning extraction...please standby')
        extractionDir = os.path.join(os.getcwd(), os.path.basename(self.filePath) + '_extracted')

        if not os.path.exists(extractionDir):
            os.mkdir(extractionDir)

        os.chdir(extractionDir)

        for entry in self.tocList:
            self.fPtr.seek(entry.position, os.SEEK_SET)
            data = self.fPtr.read(entry.cmprsdDataSize)

            if entry.cmprsFlag == 1:
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    print('[!] Error : Failed to decompress {0}'.format(entry.name))
                    continue
                assert len(data) == entry.uncmprsdDataSize

            if entry.typeCmprsData == b'd' or entry.typeCmprsData == b'o':
                continue

            basePath = os.path.dirname(entry.name)
            if basePath != '':
                if not os.path.exists(basePath):
                    os.makedirs(basePath)

            if entry.typeCmprsData == b's':
                print('[+] Possible entry point: {0}.pyc'.format(entry.name))
                if self.pycMagic == b'\0' * 4:
                    self.barePycList.append(entry.name + '.pyc')
                self._writePyc(entry.name + '.pyc', data)

            elif entry.typeCmprsData == b'M' or entry.typeCmprsData == b'm':
                if data[2:4] == b'\r\n':
                    if self.pycMagic == b'\0' * 4:
                        self.pycMagic = data[0:4]
                    self._writeRawData(entry.name + '.pyc', data)
                else:
                    if self.pycMagic == b'\0' * 4:
                        self.barePycList.append(entry.name + '.pyc')
                    self._writePyc(entry.name + '.pyc', data)

            else:
                self._writeRawData(entry.name, data)
                if entry.typeCmprsData == b'z' or entry.typeCmprsData == b'Z':
                    self._extractPyz(entry.name)

        self._fixBarePycs()

    def _fixBarePycs(self):
        for pycFile in self.barePycList:
            with open(pycFile, 'r+b') as pycFile:
                pycFile.write(self.pycMagic)

    def _writePyc(self, filename, data):
        with open(filename, 'wb') as pycFile:
            pycFile.write(self.pycMagic)
            if self.pymaj >= 3 and self.pymin >= 7:
                pycFile.write(b'\0' * 4)
                pycFile.write(b'\0' * 8)
            else:
                pycFile.write(b'\0' * 4)
                if self.pymaj >= 3 and self.pymin >= 3:
                    pycFile.write(b'\0' * 4)
            pycFile.write(data)

    def _extractPyz(self, name):
        dirName = name + '_extracted'
        if not os.path.exists(dirName):
            os.mkdir(dirName)

        with open(name, 'rb') as f:
            pyzMagic = f.read(4)
            assert pyzMagic == b'PYZ\0'
            pyzPycMagic = f.read(4)

            if self.pycMagic == b'\0' * 4:
                self.pycMagic = pyzPycMagic
            elif self.pycMagic != pyzPycMagic:
                self.pycMagic = pyzPycMagic
                print('[!] Warning: pyc magic of files inside PYZ archive are different from those in CArchive')

            if self.pymaj != sys.version_info.major or self.pymin != sys.version_info.minor:
                print('[!] Warning: This script is running in a different Python version than the one used to build the executable.')
                print('[!] Please run this script in Python {0}.{1} to prevent extraction errors during unmarshalling'.format(self.pymaj, self.pymin))
                print('[!] Skipping pyz extraction')
                return

            (tocPosition, ) = struct.unpack('!i', f.read(4))
            f.seek(tocPosition, os.SEEK_SET)

            try:
                toc = marshal.load(f)
            except:
                print('[!] Unmarshalling FAILED. Cannot extract {0}. Extracting remaining files.'.format(name))
                return

            print('[+] Found {0} files in PYZ archive'.format(len(toc)))

            if type(toc) == list:
                toc = dict(toc)

            for key in toc.keys():
                (ispkg, pos, length) = toc[key]
                f.seek(pos, os.SEEK_SET)
                fileName = key

                try:
                    fileName = fileName.decode('utf-8')
                except:
                    pass

                fileName = fileName.replace('..', '__').replace('.', os.path.sep)

                if ispkg == 1:
                    filePath = os.path.join(dirName, fileName, '__init__.pyc')
                else:
                    filePath = os.path.join(dirName, fileName + '.pyc')

                fileDir = os.path.dirname(filePath)
                if not os.path.exists(fileDir):
                    os.makedirs(fileDir)

                try:
                    data = f.read(length)
                    data = zlib.decompress(data)
                except:
                    print('[!] Error: Failed to decompress {0}, probably encrypted. Extracting as is.'.format(filePath))
                    open(filePath + '.encrypted', 'wb').write(data)
                else:
                    self._writePyc(filePath, data)


def main():
    if len(sys.argv) < 2:
        print('[+] Usage: pyinstxtractor.py <filename>')
    else:
        arch = PyInstArchive(sys.argv[1])
        if arch.open():
            if arch.checkFile():
                if arch.getCArchiveInfo():
                    arch.parseTOC()
                    arch.extractFiles()
                    arch.close()
                    print('[+] Successfully extracted pyinstaller archive: {0}'.format(sys.argv[1]))
                    print('')
                    print('You can now use a python decompiler on the pyc files within the extracted directory')
                    return
            arch.close()


if __name__ == '__main__':
    main()
