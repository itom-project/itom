from enum import Enum
from os import path


class ParamValidatorResult:
    def __init__(self, valid=None, message=None):
        if valid is None:
            self.__valid = []
        else:
            self.__valid = [valid]

        if message is None:
            self.__errorMessage = []
        else:
            self.__errorMessage = [message]

    def __repr__(self):
        return "ParamValidatorResult. Valid: " + str(self.isValid())

    def isValid(self):
        return not (False in self.__valid)

    def errorMessages(self):
        return self.__errorMessage

    def addResult(self, valid, message=None):
        self.__valid += valid
        if not message is None:
            self.__errorMessage += message  # message is a list

    def __add__(self, obj):
        if type(obj) != ParamValidatorResult:
            raise RuntimeError(
                "only another instance of ParamValidatorResult can be added to this instance."
            )
        self.addResult(obj.__valid, obj.__errorMessage)
        return self

    def throwExceptionIfInvalid(self):
        if self.isValid() == False:
            raise RuntimeError(
                "Invalid parameter input: " + str(self.__errorMessage)
            )


class ParamValidator:
    @staticmethod
    def checkFloat(key, value, minValue=None, maxValue=None):
        if not (isinstance(value, float) or isinstance(value, int)):
            return ParamValidatorResult(
                False,
                "parameter '{0}' is not convertable into a floating point number.".format(
                    key
                ),
            )
        if not (minValue is None) and value < minValue:
            return ParamValidatorResult(
                False,
                "parameter '{0}' must not be smaller than {1}".format(
                    key, minValue
                ),
            )
        if not (maxValue is None) and value > maxValue:
            return ParamValidatorResult(
                False,
                "parameter '{0}' must not be bigger than {1}".format(
                    key, maxValue
                ),
            )
        return ParamValidatorResult()

    @staticmethod
    def checkBool(key, value):
        if not (isinstance(value, bool)):
            return ParamValidatorResult(
                False, "parameter '{0}' is no boolean data type".format(key)
            )
        return ParamValidatorResult()

    @staticmethod
    def checkInt(key, value, minValue=None, maxValue=None):
        if not (isinstance(value, int)):
            return ParamValidatorResult(
                False,
                "parameter '{0}' is not convertable into a fixed point number.".format(
                    key
                ),
            )
        if not (minValue is None) and value < minValue:
            return ParamValidatorResult(
                False,
                "parameter '{0}' must not be smaller than {1}".format(
                    key, minValue
                ),
            )
        if not (maxValue is None) and value > maxValue:
            return ParamValidatorResult(
                False,
                "parameter '{0}' must not be bigger than {1}".format(
                    key, maxValue
                ),
            )
        return ParamValidatorResult()

    @staticmethod
    def checkFloatArray(
        key, value, minValue=None, maxValue=None, elemCountRange=None
    ):

        if type(value) != list and type(value) != tuple:
            value = [value]

        if not elemCountRange is None:
            if not (len(value) in elemCountRange):
                return ParamValidatorResult(
                    False,
                    "size of parameter '{0}' is not in given range {1}".format(
                        key, elemCountRange
                    ),
                )

        ret = ParamValidatorResult()
        c = 0
        for i in value:
            ret += ParamValidator.checkFloat(
                "{0}[{1}]".format(key, c), i, minValue, maxValue
            )
        return ret

    @staticmethod
    def checkIntArray(
        key, value, minValue=None, maxValue=None, elemCountRange=None
    ):

        if type(value) != list and type(value) != tuple:
            value = [value]

        if not elemCountRange is None:
            if not (len(value) in elemCountRange):
                return ParamValidatorResult(
                    False,
                    "size of parameter '{0}' is not in given range {1}".format(
                        key, elemCountRange
                    ),
                )

        ret = ParamValidatorResult()
        c = 0
        for i in value:
            ret += ParamValidator.checkInt(
                "{0}[{1}]".format(key, c), i, minValue, maxValue
            )
        return ret

    @staticmethod
    def checkIfInList(key, value, allowedList):
        if not value in allowedList:
            return ParamValidatorResult(
                False,
                "parameter '{0}' does not correspond to allowed values ({1})".format(
                    key, allowedList
                ),
            )
        return ParamValidatorResult()

    @staticmethod
    def checkIfInEnum(key, value, allowedEnum):
        if not (
            allowedEnum.key_exists(value) or allowedEnum.value_exists(value)
        ):
            return ParamValidatorResult(
                False,
                "parameter '{0}' does not correspond to enumeration {1}".format(
                    key, allowedEnum
                ),
            )
        return ParamValidatorResult()

    @staticmethod
    def checkIfFolderExists(key, value, allowEmptyFolder=False):
        v = str(value)
        if v == "" and allowEmptyFolder:
            return ParamValidatorResult()
        elif v == "":
            return ParamValidatorResult(False, "the given folder-name is empty")
        elif path.exists(str(value)) == False:
            return ParamValidatorResult(
                False, "the folder parameter '{0}' does not exist".format(v)
            )
        return ParamValidatorResult()
